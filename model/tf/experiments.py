import tensorflow as tf
import tensorflow_addons as tfa
import wandb

def run_experiment(model:tf.keras.Model,
                   train_generator:tf.data.Dataset,
                   val_generator:tf.data.Dataset,
                   epochs:int=10,
                   learning_rate:float=0.00005,
                   weight_decay:float=0.0001,
                   label_smoothing:float=.1,
                   checkpoint_filepath:str="/content/tmp/checkpoint",
                   verbose:int=1,
                   ):
    # Check for GPU:
    assert len(tf.config.list_physical_devices('GPU'))>0, f"No GPU available. Check system settings."

    # Train & validation steps
    train_steps_per_epoch = len(train_generator)
    val_steps_per_epoch = len(val_generator)

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
    checkpoint_filepath = checkpoint_filepath
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
    )

    # Model fit
    history = model.fit(
        train_generator,
        steps_per_epoch= train_steps_per_epoch,
        epochs = epochs,
        validation_data=val_generator,
        validation_steps = val_steps_per_epoch,
        callbacks=[checkpoint_callback],
        verbose = verbose,
    )
    # Clear memory
    tf.keras.backend.clear_session()

    return history


def run_WB_experiment(WB_KEY:str,
                      WB_ENTITY:str,
                      WB_PROJECT:str,
                      model:tf.keras.Model,
                      train_generator:tf.data.Dataset,
                      val_generator:tf.data.Dataset,
                      epochs:int=10,
                      learning_rate:float=0.00005,
                      weight_decay:float=0.0001,
                      label_smoothing:float=.1,
                      checkpoint_filepath:str="/content/tmp/checkpoint",
                      verbose:int=1,
                      ):
    # Check for GPU:
    assert len(tf.config.list_physical_devices('GPU'))>0, f"No GPU available. Check system settings."
    # Credentials
    wandb.login(key=WB_KEY)
    wandb.init(project=WB_PROJECT, entity=WB_ENTITY)
 
    # Train & validation steps
    train_steps_per_epoch = len(train_generator)
    val_steps_per_epoch = len(val_generator)

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
    checkpoint_filepath = checkpoint_filepath
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
    )
    wandb_callback = wandb.keras.WandbCallback()

    # Model fit
    history = model.fit(
        train_generator,
        steps_per_epoch= train_steps_per_epoch,
        epochs = epochs,
        validation_data=val_generator,
        validation_steps = val_steps_per_epoch,
        callbacks=[checkpoint_callback, wandb_callback],
        verbose = verbose,
    )
    # Clear memory
    tf.keras.backend.clear_session()
    
    return history