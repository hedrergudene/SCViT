import tensorflow as tf
import tensorflow_addons as tfa
import wandb
from typing import Dict, List
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from hvit.medmnist.info import INFO
from hvit.tf.custom_metrics import f1
import hvit.medmnist.dataset_without_pytorch as mdn
import cv2

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def run_WB_experiment(WB_KEY:str,
                      WB_ENTITY:str,
                      WB_PROJECT:str,
                      WB_GROUP:str,
                      model:tf.keras.Model,
                      data_flag:str,
                      ImageDataGenerator_config:Dict,
                      flow_config:Dict,
                      epochs:int=10,
                      learning_rate:float=0.00005,
                      weight_decay:float=0.0001,
                      label_smoothing:float=.1,
                      es_patience:int=10,
                      verbose:int=1,
                      resize:int = None,
                      ):
    # Check for GPU:
    assert len(tf.config.list_physical_devices('GPU'))>0, f"No GPU available. Check system settings."

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(mdn, info['python_class'])
    print(f'Dataset {data_flag} Task {task} n_channels {n_channels} n_classes {n_classes}')

    # load train Data
    train_dataset = DataClass(split='train', download=True)
    x_train = train_dataset.imgs
    if resize is not None:
        x_train = np.stack([cv2.resize(img, (resize,resize), interpolation = cv2.INTER_AREA) for img in x_train])
    y_train = train_dataset.labels
    print(f'X train {x_train.shape} | Y train {y_train.shape}')  

    # load val Data
    train_dataset = DataClass(split='val', download=True)
    x_val = train_dataset.imgs
    if resize is not None:
        x_val = np.stack([cv2.resize(img, (resize,resize), interpolation = cv2.INTER_AREA) for img in x_val])
    y_val = train_dataset.labels
    print(f'X train {x_val.shape} | Y train {y_val.shape}')

    # load test Data
    train_dataset = DataClass(split='test', download=True)
    x_test = train_dataset.imgs
    if resize is not None:
        x_test = np.stack([cv2.resize(img, (resize,resize), interpolation = cv2.INTER_AREA) for img in x_test])
    y_test = train_dataset.labels
    print(f'X train {x_test.shape} | Y train {y_test.shape}')

    # Log in WB
    wandb.login(key=WB_KEY)
    # Generators
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=ImageDataGenerator_config['train']['rescale'],
                                                                    shear_range=ImageDataGenerator_config['train']['shear_range'],
                                                                    rotation_range=ImageDataGenerator_config['train']['rotation_range'],
                                                                    zoom_range=ImageDataGenerator_config['train']['zoom_range'],
                                                                    horizontal_flip=ImageDataGenerator_config['train']['horizontal_flip'],
                                                                    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=ImageDataGenerator_config['val']['rescale'])
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=ImageDataGenerator_config['test']['rescale'])
    train_generator = train_datagen.flow(x=x_train, 
                                         y=y_train,
                                         batch_size=flow_config['train']['batch_size'],
                                         shuffle=flow_config['train']['shuffle'],
                                         seed=flow_config['train']['seed'],
                                         )
    val_generator = val_datagen.flow(x=x_val,
                                     y=y_val,
                                     batch_size=flow_config['val']['batch_size'],
                                     shuffle=flow_config['val']['shuffle'],
                                     seed=flow_config['val']['seed'],
                                     )
    test_generator = test_datagen.flow(x=x_test,
                                       y=y_test,
                                       batch_size=flow_config['test']['batch_size'],
                                       shuffle=flow_config['test']['shuffle'],
                                       seed=flow_config['test']['seed'],
                                       )
    # Train & validation steps
    train_steps_per_epoch = len(train_generator)
    val_steps_per_epoch = len(val_generator)
    test_steps_per_epoch = len(test_generator)

    # Save initial weights
    #model.load_weights(os.path.join(os.getcwd(), 'model_weights.h5'))

    # Credentials
    wandb.init(project=WB_PROJECT, entity=WB_ENTITY, group = WB_GROUP)
    
    # Model compile
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    if task == 'multi-class':
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['accuracy',f1]
    if task == 'binary-class':
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing = label_smoothing)
        metrics = ['accuracy','auc']

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    # Callbacks
    reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=learning_rate//10, verbose=1)
    patience = tf.keras.callbacks.EarlyStopping(patience=es_patience)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(os.getcwd(), 'model_best_weights.h5'), save_best_only = True, save_weights_only = True)
    wandb_callback = wandb.keras.WandbCallback(save_weights_only=True)
    # Model fit
    history = model.fit(
        train_generator,
        steps_per_epoch= train_steps_per_epoch,
        epochs = epochs,
        validation_data=val_generator,
        validation_steps = val_steps_per_epoch,
        callbacks=[reduceLR, patience, checkpoint, wandb_callback],
        verbose = verbose,
    )

    # Evaluation
    model.load_weights(os.path.join(os.getcwd(), 'model_best_weights.h5'))
    results = model.evaluate(test_generator, steps = test_steps_per_epoch, verbose = 0)
    print("Test metrics:",{k:v for k,v in zip(model.metrics_names, results)})
    wandb.log({("test_"+k):v for k,v in zip(model.metrics_names, results)})

    # Clear memory
    tf.keras.backend.clear_session()
    wandb.finish()


def run_WB_CV_experiment(WB_KEY:str,
                      WB_ENTITY:str,
                      WB_PROJECT:str,
                      WB_GROUP:str,
                      model:tf.keras.Model,
                      ImageDataGenerator_config:Dict,
                      flow_from_dataframe_config:Dict,
                      path:str="/content/OCT2017 /train/",
                      test_path:str="/content/OCT2017 /test/",
                      folds:int=4,
                      epochs:int=8,
                      learning_rate:float=0.00005,
                      weight_decay:float=0.0001,
                      es_patience:int=10,
                      label_smoothing:float=.1,
                      seed:int=123,
                      verbose:int=1,
                      ):
    # Check for GPU:
    assert len(tf.config.list_physical_devices('GPU'))>0, f"No GPU available. Check system settings."
    # Take initial model weights to reset each iteration
    model.save_weights(os.path.join(os.getcwd(), 'model_weights.h5'))
    # Log in WB
    wandb.login(key=WB_KEY)
    # Set up cross validation
    df = get_df(path)
    kf = StratifiedKFold(n_splits = folds, shuffle = True, random_state = seed)
    for i, (train_idx, val_idx) in enumerate(kf.split(df['x_col'], df['y_col'])):
        # Gather data
        train_df = df.iloc[train_idx,:]
        val_df = df.iloc[val_idx,:]
        test_df = get_df(test_path)
        # Generators
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=ImageDataGenerator_config['train']['rescale'],
                                                                        shear_range=ImageDataGenerator_config['train']['shear_range'],
                                                                        rotation_range=ImageDataGenerator_config['train']['rotation_range'],
                                                                        zoom_range=ImageDataGenerator_config['train']['zoom_range'],
                                                                        horizontal_flip=ImageDataGenerator_config['train']['horizontal_flip'],
                                                                        )
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=ImageDataGenerator_config['val']['rescale'])
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=ImageDataGenerator_config['test']['rescale'])
        flow_from_dataframe_config['train']['dataframe'] = train_df
        flow_from_dataframe_config['val']['dataframe'] = val_df
        flow_from_dataframe_config['test']['dataframe'] = test_df
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
        test_generator = test_datagen.flow_from_dataframe(dataframe=flow_from_dataframe_config['test']['dataframe'],
                                      x_col=flow_from_dataframe_config['test']['x_col'],
                                      y_col=flow_from_dataframe_config['test']['y_col'],
                                      target_size=flow_from_dataframe_config['test']['target_size'],
                                      batch_size=flow_from_dataframe_config['test']['batch_size'],
                                      color_mode=flow_from_dataframe_config['test']['color_mode'],
                                      class_mode=flow_from_dataframe_config['test']['class_mode'],
                                      shuffle=flow_from_dataframe_config['test']['shuffle'],
                                      seed=flow_from_dataframe_config['test']['seed'],
                                      )
        # Train & validation steps
        train_steps_per_epoch = len(train_generator)
        val_steps_per_epoch = len(val_generator)
        test_steps_per_epoch = len(test_generator)
        # Save initial weights
        model.load_weights(os.path.join(os.getcwd(), 'model_weights.h5'))
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
                f1,
            ],
        )
        # Callbacks
        reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=learning_rate//10, verbose=1)
        patience = tf.keras.callbacks.EarlyStopping(patience=es_patience)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(os.getcwd(), 'model_best_weights.h5'), save_best_only = True, save_weights_only = True)
        wandb_callback = wandb.keras.WandbCallback(save_weights_only=True)
        # Model fit
        history = model.fit(
            train_generator,
            steps_per_epoch= train_steps_per_epoch,
            epochs = epochs,
            validation_data=val_generator,
            validation_steps = val_steps_per_epoch,
            callbacks=[reduceLR, patience, checkpoint, wandb_callback],
            verbose = verbose,
        )
        # Evaluation
        model.load_weights(os.path.join(os.getcwd(), 'model_best_weights.h5'))
        results = model.evaluate(test_generator, steps = test_steps_per_epoch, verbose = 0)
        print("Test metrics:",{k:v for k,v in zip(model.metrics_names, results)})
        wandb.log({'test_loss':results[0], 'test_accuracy':results[1]})
        # Clear memory
        tf.keras.backend.clear_session()
        wandb.finish()
