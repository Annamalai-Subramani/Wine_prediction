def some_function():
    from Wine_prediction.config.configuration import ConfigurationManager
    a = ConfigurationManager()
    a.get_data_ingestion_config()

if __name__ == "__main__":
    some_function()