import pandas as pd

def load_and_merge():
    # LOAD DATASETS
    orders = pd.read_csv("data/orders.csv", encoding="latin1")
    payments = pd.read_csv("data/payments.csv", encoding="latin1")
    stores = pd.read_csv("data/stores.csv", encoding="latin1")
    deliveries = pd.read_csv("data/deliveries.csv", encoding="latin1")
    drivers = pd.read_csv("data/drivers.csv", encoding="latin1")
    channels = pd.read_csv("data/channels.csv", encoding="latin1")
    hubs = pd.read_csv("data/hubs.csv", encoding="latin1")

    # MERGE DATASETS
    df = orders.merge(payments, on="payment_order_id", how="left")
    df = df.merge(stores, on="store_id", how="left")
    df = df.merge(deliveries, on="delivery_order_id", how="left")
    df = df.merge(drivers, on="driver_id", how="left")
    df = df.merge(channels, on="channel_id", how="left")
    df = df.merge(hubs, on="hub_id", how="left")

    print("Shape do dataset após o merge:", df.shape)
    return df