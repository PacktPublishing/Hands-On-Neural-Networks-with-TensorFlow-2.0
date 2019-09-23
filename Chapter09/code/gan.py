import tensorflow as tf
import matplotlib.pyplot as plt


def sample_dataset():
    dataset_shape = (2000, 1)
    return tf.random.normal(mean=10., shape=dataset_shape, stddev=0.1, dtype=tf.float32)


counts, bin, ignored = plt.hist(sample_dataset().numpy(), 100)
axes = plt.gca()
axes.set_xlim([-1,11])
axes.set_ylim([0, 60])
plt.show()

def generator(input_shape):
    """Defines the generator keras.Model.
    Args:
        input_shape: the desired input shape (e.g.: (latent_space_size))
    Returns:
        G: The generator model
    """
    inputs = tf.keras.layers.Input(input_shape)
    net = tf.keras.layers.Dense(units=64, activation=tf.nn.elu, name="fc1")(inputs)
    net = tf.keras.layers.Dense(units=64, activation=tf.nn.elu, name="fc2")(net)
    net = tf.keras.layers.Dense(units=1, name="G")(net)
    G = tf.keras.Model(inputs=inputs, outputs=net)
    return G

def disciminator(input_shape):
    """Defines the Discriminator keras.Model.
    Args:
        input_shape: the desired input shape (e.g.: (the generator output shape))
    Returns:
        D: the Discriminator model
    """
    inputs = tf.keras.layers.Input(input_shape)
    net = tf.keras.layers.Dense(units=32, activation=tf.nn.elu, name="fc1")(inputs)
    net = tf.keras.layers.Dense(units=1, name="D")(net)
    D = tf.keras.Model(inputs=inputs, outputs=net)
    return D

# Define the real input shape
input_shape = (1,)

# Define the Discriminator model
D = disciminator(input_shape)

# Arbitrary set the shape of the noise prior
latent_space_shape = (100,)
# Define the input noise shape and define the generator
G = generator(latent_space_shape)

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def d_loss(d_real, d_fake):
    """The disciminator loss function."""
    return bce(tf.ones_like(d_real), d_real) + bce(tf.zeros_like(d_fake), d_fake)

def train():
    # Define the optimizers and the train operations
    optimizer = tf.keras.optimizers.Adam(1e-5)
    
    @tf.function
    def train_step():
        with tf.GradientTape(persistent=True) as tape:
            real_data = sample_dataset()
            noise_vector = tf.random.normal(
                mean=0, stddev=1,
                shape=(real_data.shape[0], latent_space_shape[0]))
            # Sample from the Generator
            fake_data = G(noise_vector)
            # Compute the D loss
            d_fake_data = D(fake_data)
            d_real_data = D(real_data)
            d_loss_value = d_loss(d_real_data, d_fake_data)
            # Compute the G loss
            g_loss_value = g_loss(d_fake_data)
        # Now that we comptuted the losses we can compute the gradient
        # and optimize the networks
        d_gradients = tape.gradient(d_loss_value, D.trainable_variables)
        g_gradients = tape.gradient(g_loss_value, G.trainable_variables)
        # Deletng the tape, since we defined it as persistent
        # (because we used it twice)
        del tape
        
        optimizer.apply_gradients(zip(d_gradients, D.trainable_variables))
        optimizer.apply_gradients(zip(g_gradients, G.trainable_variables))
        return real_data, fake_data, g_loss_value, d_loss_value
    

    fig, ax = plt.subplots()
    for step in range(40000):
        real_data, fake_data,g_loss_value, d_loss_value = train_step()
        if step % 200 == 0:
            print("G loss: ", g_loss_value.numpy(), " D loss: ", d_loss_value.numpy(), " step: ", step)

            # Sample 5000 values from the Generator and draw the histogram
            ax.hist(fake_data.numpy(), 100)
            ax.hist(real_data.numpy(), 100)
            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            # place a text box in upper left in axes coords
            textstr = f"step={step}"
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)

            axes = plt.gca()
            axes.set_xlim([-1,11])
            axes.set_ylim([0, 60])
            display.display(pl.gcf())
            display.clear_output(wait=True)
            plt.gca().clear()
            
train()            
