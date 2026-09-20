from .spark_studio_node import SparkStudioChat

NODE_CLASS_MAPPINGS = {"SparkStudioChat": SparkStudioChat}
NODE_DISPLAY_NAME_MAPPINGS = {"SparkStudioChat": "Spark Chat (OpenAI-compatible)"}
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
