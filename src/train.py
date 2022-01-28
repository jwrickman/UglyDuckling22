import pytorch_lightning as pl
import glob

from pl_bolts.models.autoencoders import VAE

from pipeline import MoleMapDataModule

DATA_DIR = "/media/storage4/molemap/dumped_images/"


if __name__ == "__main__":

    image_paths = glob.glob(DATA_DIR + "*.jpg")
    print(len(image_paths))

    datamodule = MoleMapDataModule(
        image_paths=image_paths,
        batch_size=32,
        image_size=128,
        num_workers=6,
        persistent_workers=False)


    model = VAE(
        input_height=128,
        enc_type="resnet18",
        latent_dim=12
    )


    ### Train ###
    trainer = pl.Trainer(
        accelerator="gpu",
        gpus=1,
        max_epochs=5,
        auto_scale_batch_size=True,
    )

    trainer.fit(model, datamodule)
