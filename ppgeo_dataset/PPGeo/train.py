import argparse
import os
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from model import Monodepth, MotionNet
from data_ytb import YTB_Data

class PPGeoEngine(pl.LightningModule):
	def __init__(self, config):
		super().__init__()
		self.stage = config.stage
		assert self.stage in [1,2]
		self.lr = config.lr
		self.config = config
		self.stage = self.stage
		self.model = Monodepth(stage = self.stage, batch_size=config.batch_size)
		if self.stage == 2:

			path_to_depth = "ppgeo_depth.ckpt"
			print(f"Loading depth nets from for stage 2 from {path_to_depth}")
			ckpt = torch.load(path_to_depth, map_location="cpu")

			# Expected PPGeo stage-1 export with two sub dicts
			enc_sd = ckpt.get("depth_encoder_state_dict", {})

			dec_sd = ckpt.get("depth_decoder_state_dict", {})

			missing, unexpected = self.model.depth_encoder.load_state_dict(enc_sd, strict = False)
			print(f"depth  encoder Missing keys: {len(missing)}  Unexpected: {len(unexpected)}")

			enc_keys_loaded = set(enc_sd.keys())
			enc_keys_model = set(self.model.depth_encoder.state_dict().keys())
			print(f"[Debug] Encoder: Loaded {len(enc_keys_loaded)} keys, Model expects {len(enc_keys_model)} keys")
			print(f"[Debug] Encoder: Intersection {len(enc_keys_loaded & enc_keys_model)} keys")

			# Check a few example layers for identical shape
			for k in list(enc_keys_loaded & enc_keys_model)[:5]:
				w_loaded = enc_sd[k].shape
				w_model = self.model.depth_encoder.state_dict()[k].shape
				print(f"[Debug] Encoder layer '{k}': loaded {w_loaded}, model {w_model}")

			missing, unexpected = self.model.depth_decoder.load_state_dict(dec_sd, strict = False)
			print(f"depth decoder Missing keys: {len(missing)}  Unexpected: {len(unexpected)}")

			# === Optional: warm-start pose nets too (only if present & compatible) ===
			path_to_pose = "ppgeo_pose.ckpt"

			print(f"[stage 2 Loading pose nets from {path_to_pose}")
			ckpt_pose = torch.load(path_to_pose, map_location="cpu")


			pose_enc_sd = ckpt_pose.get("pose_encoder_state_dict", {})
			pose_dec_sd = ckpt_pose.get("pose_decoder_state_dict", {})

			# Some exports include a classification head in the encoder (fc.*); drop it
			if pose_enc_sd:
				pose_enc_sd = {k: v for k, v in pose_enc_sd.items()
							if not (k.startswith("encoder.fc.") or k == "encoder.fc.weight" or k == "encoder.fc.bias")}

			# --- load pose encoder if present ---
			if pose_enc_sd:
				pe_missing, pe_unexp = self.model.pose_encoder.load_state_dict(pose_enc_sd, strict=False)
				print(f" pose-encoder Missing: {len(pe_missing)}  Unexpected: {len(pe_unexp)}")
				if pe_missing:
					print(" pose-encoder missing keys:", pe_missing)

				# sanity: first conv should be 6x7x7 for 2 frames of RGB (or gray→RGB)
				try:
					k = "encoder.conv1.weight"
					w_ckpt  = tuple(pose_enc_sd[k].shape)
					w_model = tuple(self.model.pose_encoder.state_dict()[k].shape)
					print(f"[Debug] PoseEnc '{k}': loaded {w_ckpt} model {w_model}")
				except KeyError:
					pass
			else:
				print(" No pose-encoder weights in ckpt.")

			# --- load pose decoder if present ---
			if pose_dec_sd:
				pd_missing, pd_unexp = self.model.pose_decoder.load_state_dict(pose_dec_sd, strict=False)
				print(f" pose-decoder Missing: {len(pd_missing)}  Unexpected: {len(pd_unexp)}")
			else:
				print("  No pose-decoder weights in ckpt.")

			self.motionnet = MotionNet()
			self.model.eval()
			for param in self.model.parameters():
				param.requires_grad = False
	
	def forward(self, batch):
		pass

	def training_step(self, batch, batch_idx):
		if self.stage == 1:
			outputs, losses = self.model(batch)
		else:
			self.model.eval()
			motion = self.motionnet(batch)
			outputs, losses = self.model(batch, *motion)
		for k,v in losses.items():
			self.log('train_{}'.format(k), v.item())

		return losses['loss']

	def configure_optimizers(self):
		if self.stage == 2:
			optimizer = optim.AdamW(self.motionnet.parameters(), lr=self.lr, weight_decay=1e-4)
		else:
			optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
		lr_scheduler = optim.lr_scheduler.CyclicLR(
			optimizer, base_lr=1e-6, max_lr=1e-4, step_size_up=2000, cycle_momentum=False)
		return [optimizer], [lr_scheduler]


	def validation_step(self, batch, batch_idx):
		if self.stage == 1:
			outputs, losses = self.model(batch)
		else:
			motion = self.motionnet(batch)
			outputs, losses = self.model(batch, *motion)

		for k,v in losses.items():
			self.log('val_{}'.format(k), v.item(), sync_dist=True)

		self.log("val_loss", losses['loss/0'].item(), sync_dist=True)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--id', type=str, default='ppgeo_stage1_log', help='Unique experiment identifier.')
	parser.add_argument('--stage', type=int, default=1, help='stage 1 for depth and pose networks, stage 2 for visual encoder')
	parser.add_argument('--ckpt', type=str, help='stage 1 ckpt')
	parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
	parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
	parser.add_argument('--val_every', type=int, default=3, help='Validation frequency (epochs).')
	parser.add_argument('--batch_size', type=int, default=48, help='Batch size')
	parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')

	args = parser.parse_args()
	args.logdir = os.path.join(args.logdir, args.id)

	train_set = YTB_Data(root="data", meta_path = "ytb_meta_train_trip.npy", is_train=True)
	val_set = YTB_Data(root="data", meta_path = "ytb_meta_val_trip.npy", is_train=False)
	print(len(train_set))
	print(len(val_set))

	dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
	dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)

	ppgeo = PPGeoEngine(args)

	checkpoint_callback = ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_loss", save_top_k=1, save_last=True,
											dirpath=args.logdir, filename="best_{epoch:02d}-{val_loss:.3f}")
	checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
	trainer = pl.Trainer.from_argparse_args(args,
											default_root_dir=args.logdir,
											gpus = 1,
											accelerator='ddp',
											sync_batchnorm=True,
											plugins=DDPPlugin(find_unused_parameters=True),
											profiler='simple',
											benchmark=True,
											log_every_n_steps=1,
											flush_logs_every_n_steps=5,
											callbacks=[checkpoint_callback,
														],
											check_val_every_n_epoch = 3,
											max_epochs = args.epochs
											)

	trainer.fit(ppgeo, dataloader_train, dataloader_val)




		