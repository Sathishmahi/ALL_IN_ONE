from exception import CustomException
import sys
def abc(a,b):
    try:
        return a/b
    except:
        raise CustomException(sys)
