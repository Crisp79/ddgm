import torch
from torch .utils .data import Dataset ,DataLoader ,Subset
from torchvision import datasets ,transforms
import os
import matplotlib .pyplot as plt





def get_transforms (image_size =64 ):
    return transforms .Compose ([
    transforms .CenterCrop (178 ),
    transforms .Resize ((64 ,64 )),
    transforms .RandomHorizontalFlip (),
    transforms .ToTensor (),
    transforms .Normalize ([0.5 ]*3 ,[0.5 ]*3 )
    ])





def load_full_dataset (data_path ,image_size =64 ):
    transform =get_transforms (image_size )

    dataset =datasets .ImageFolder (
    root =data_path ,
    transform =transform
    )

    return dataset





def get_subset (dataset ,num_samples =20000 ):
    assert num_samples <=len (dataset ),"Subset size larger than dataset"

    indices =torch .randperm (len (dataset ))[:num_samples ]
    subset =Subset (dataset ,indices )

    return subset





def get_dataloader (dataset ,batch_size =32 ,num_workers =4 ):
    loader =DataLoader (
    dataset ,
    batch_size =batch_size ,
    shuffle =True ,
    num_workers =num_workers ,
    pin_memory =True
    )

    return loader





def get_celeba_loader (
data_path ,
image_size =64 ,
batch_size =32 ,
subset_size =20000 ,
num_workers =4
):
    dataset =load_full_dataset (data_path ,image_size )

    if subset_size is not None :
        dataset =get_subset (dataset ,subset_size )

    loader =get_dataloader (dataset ,batch_size ,num_workers )

    return loader





if __name__ =="__main__":
    data_path ="data/celeba"

    loader =get_celeba_loader (data_path ,subset_size =2 )

    images ,_ =next (iter (loader ))

    img =images [0 ].permute (1 ,2 ,0 )
    img =(img +1 )/2

    plt .imshow (img )
    plt .axis ("off")
    plt .show ()

    print ("Shape:",images .shape )