from src.utils.logging_config import setup_logging, CustomLogger


def main():
    logger: CustomLogger = setup_logging(full_color=True, include_function=True)  # type: ignore
    logger.success("Logging configuration test completed.")


if __name__ == "__main__":
    main()
