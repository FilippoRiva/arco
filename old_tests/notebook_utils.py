def load_secrets_from_keyring(search_attributes:dict):
    import secretstorage

    # 1. Connect to the D-Bus secret service
    connection = secretstorage.dbus_init()
    collection = secretstorage.get_default_collection(connection)

    items = list(collection.search_items(search_attributes))

    if items:
        # Get the secret from the first matching item
        return items[0].get_secret().decode('utf-8')
    return None