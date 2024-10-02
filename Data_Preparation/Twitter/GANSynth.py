import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('twittersource.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%b-%y')
data.set_index('Date', inplace=True)

# Select the required columns for war/conflict data
war_columns = [
    'WAR/CONFLICT US', 'WAR/CONFLICT GB', 'WAR/CONFLICT CA', 'WAR/CONFLICT AU', 'WAR/CONFLICT UA',
    'WAR/CONFLICT RU', 'WAR/CONFLICT FR', 'WAR/CONFLICT DE', 'WAR/CONFLICT BR', 'WAR/CONFLICT CN',
    'WAR/CONFLICT JP', 'WAR/CONFLICT PK', 'WAR/CONFLICT KP', 'WAR/CONFLICT KR', 'WAR/CONFLICT IN',
    'WAR/CONFLICT TW', 'WAR/CONFLICT NL', 'WAR/CONFLICT ES', 'WAR/CONFLICT SE', 'WAR/CONFLICT MX',
    'WAR/CONFLICT IR', 'WAR/CONFLICT IL', 'WAR/CONFLICT SA', 'WAR/CONFLICT SY', 'WAR/CONFLICT FI',
    'WAR/CONFLICT IE', 'WAR/CONFLICT AT', 'WAR/CONFLICT NO', 'WAR/CONFLICT CH', 'WAR/CONFLICT IT',
    'WAR/CONFLICT MY', 'WAR/CONFLICT EG', 'WAR/CONFLICT TR', 'WAR/CONFLICT PT', 'WAR/CONFLICT PS',
    'WAR/CONFLICT AE', 'WAR/CONFLICT ALL'
]

# Preprocess data (scaling)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[war_columns])

# Set parameters for GAN
latent_dim = 100
data_shape = scaled_data.shape[1]  # Number of columns (features)

# Generator model
def build_generator(latent_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(output_dim, activation='tanh'))
    model.add(Reshape((output_dim,)))
    return model

# Discriminator model
def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(1, activation='sigmoid'))
    model.add(Flatten())
    return model

# Compile GAN models
def build_gan(generator, discriminator):
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    generated_data = generator(gan_input)
    gan_output = discriminator(generated_data)
    gan = Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    return gan

# GAN training
def train_gan(generator, discriminator, gan, data, latent_dim, epochs=5000, batch_size=64, sample_interval=1000):
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        # Training the discriminator
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_data = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Training the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real_labels)
        
        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D acc.: {100*d_loss[1]}] [G loss: {g_loss}]")
            sample_generated_data(generator, epoch, latent_dim)

# Sample and save generated data
def sample_generated_data(generator, epoch, latent_dim):
    noise = np.random.normal(0, 1, (1, latent_dim))
    gen_data = generator.predict(noise)
    scaled_gen_data = scaler.inverse_transform(gen_data)
    
    plt.figure(figsize=(10, 6))
    plt.plot(scaled_gen_data.flatten(), label='Generated Data')
    plt.title(f'Generated War/Conflict Data at Epoch {epoch}')
    plt.legend()
    plt.show()

# Build and compile the models
generator = build_generator(latent_dim, data_shape)
discriminator = build_discriminator(data_shape)
gan = build_gan(generator, discriminator)

# Train the GAN
train_gan(generator, discriminator, gan, scaled_data, latent_dim, epochs=10000, batch_size=64, sample_interval=1000)

# After training, generate synthetic data for the future period (Jan 2023 - Apr 2024)
future_data = []
for i in range(16):  # For 16 months from Jan 2023 to Apr 2024
    noise = np.random.normal(0, 1, (1, latent_dim))
    generated_data = generator.predict(noise)
    scaled_gen_data = scaler.inverse_transform(generated_data)
    future_data.append(scaled_gen_data.flatten())

# Convert generated future data to DataFrame
future_dates = pd.date_range(start='2023-01-01', periods=16, freq='M')
future_df = pd.DataFrame(future_data, index=future_dates, columns=war_columns)

# Save generated future data to CSV
future_df.to_csv('twitterfinaldata.csv')
print("Generated data saved to 'GeneratedWarConflictData_Jan2023_Apr2024.csv'")
