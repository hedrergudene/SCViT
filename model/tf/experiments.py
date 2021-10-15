import tensorflow as tf
import tensorflow_addons as tfa
import wandb
from sklearn.model_selection import KFold
from typing import Dict, List
import os
import pandas as pd

def get_df(path:str):
    labels = os.listdir(path)
    df = pd.DataFrame({'x_col':[], 'y_col':[]})
    for label in labels:
        group = [path + "/" + label + "/" + filename for filename in os.listdir(os.path.join(path,label))]
        df = pd.concat([df, pd.DataFrame({'x_col':group, 'y_col':[label]*len(group)})], axis = 0)
    return df


def run_WB_experiment(WB_KEY:str,
                      WB_ENTITY:str,
                      WB_PROJECT:str,
                      WB_GROUP:str,
                      model:tf.keras.Model,
                      ImageDataGenerator_config:Dict,
                      flow_from_dataframe_config:Dict,
                      path:str="/content/OCT2017 /",
                      epochs:int=10,
                      folds:int=5,
                      learning_rate:float=0.00005,
                      weight_decay:float=0.0001,
                      label_smoothing:float=.1,
                      seed:int=123,
                      verbose:int=1,
                      ):
    # Check for GPU:
    assert len(tf.config.list_physical_devices('GPU'))>0, f"No GPU available. Check system settings."

    # Gather data
    df = get_df(path)
    kf = KFold(n_splits = folds, shuffle = True, random_state = seed)
    # Log in WB
    wandb.login(key=WB_KEY)
    # Start X-validation
    for train_idx, val_idx in kf.split(df):
        # Generators
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=ImageDataGenerator_config['train']['rescale'],
                                                                        shear_range=ImageDataGenerator_config['train']['shear_range'],
                                                                        rotation_range=ImageDataGenerator_config['train']['rotation_range'],
                                                                        zoom_range=ImageDataGenerator_config['train']['zoom_range'],
                                                                        horizontal_flip=ImageDataGenerator_config['train']['horizontal_flip'],
                                                                        )
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=ImageDataGenerator_config['val']['rescale'])
        flow_from_dataframe_config['train']['dataframe'] = df.iloc[train_idx]
        flow_from_dataframe_config['val']['dataframe'] = df.iloc[val_idx]
        train_generator = train_datagen.flow_from_dataframe(dataframe=flow_from_dataframe_config['train']['dataframe'],
                                      x_col=flow_from_dataframe_config['train']['x_col'],
                                      y_col=flow_from_dataframe_config['train']['y_col'],
                                      target_size=flow_from_dataframe_config['train']['target_size'],
                                      batch_size=flow_from_dataframe_config['train']['batch_size'],
                                      color_mode=flow_from_dataframe_config['train']['color_mode'],
                                      class_mode=flow_from_dataframe_config['train']['class_mode'],
                                      shuffle=flow_from_dataframe_config['train']['shuffle'],
                                      seed=flow_from_dataframe_config['train']['seed'],
                                      )
        val_generator = val_datagen.flow_from_dataframe(dataframe=flow_from_dataframe_config['val']['dataframe'],
                                      x_col=flow_from_dataframe_config['val']['x_col'],
                                      y_col=flow_from_dataframe_config['val']['y_col'],
                                      target_size=flow_from_dataframe_config['val']['target_size'],
                                      batch_size=flow_from_dataframe_config['val']['batch_size'],
                                      color_mode=flow_from_dataframe_config['val']['color_mode'],
                                      class_mode=flow_from_dataframe_config['val']['class_mode'],
                                      shuffle=flow_from_dataframe_config['val']['shuffle'],
                                      seed=flow_from_dataframe_config['val']['seed'],
                                      )
        # Train & validation steps
        train_steps_per_epoch = len(train_generator)
        val_steps_per_epoch = len(val_generator)
        # Credentials
        wandb.init(project=WB_PROJECT, entity=WB_ENTITY, group = WB_GROUP)
        # Model compile
        optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing = label_smoothing),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            ],
        )
        # Callbacks
        reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=learning_rate//10)
        patience = tf.keras.callbacks.EarlyStopping(patience=2),
        wandb_callback = wandb.keras.WandbCallback()
        # Model fit
        history = model.fit(
            train_generator,
            steps_per_epoch= train_steps_per_epoch,
            epochs = epochs,
            validation_data=val_generator,
            validation_steps = val_steps_per_epoch,
            callbacks=[reduceLR, patience, wandb_callback],
            verbose = verbose,
        )
        # Clear memory
        tf.keras.backend.clear_session()
