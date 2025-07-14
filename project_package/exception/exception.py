import sys
from project_package.logging import logger

class ProjectException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(str(error_message))
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()
        self.lineno = exc_tb.tb_lineno if exc_tb else None
        self.file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else None

    def __str__(self):
        return f"Error occurred in script [{self.file_name}] at line [{self.lineno}]: {self.error_message}"

        
if __name__=='__main__':
    try:
        logger.logging.info("Enter the try block for checking exception handling")
        a=1/0
        print("This info will not be printed", a)
    except Exception as e:
           raise ProjectException(e, sys)