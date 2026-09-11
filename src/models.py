from enum import Enum
from pydantic import BaseModel


class Category(str, Enum):
    HARDWARE = "Hardware"
    SOFTWARE = "Software"
    NETWORK = "Netwerk"
    ACCESS_ACCOUNTS = "Toegang & Accounts"
    PRINTER = "Printer"
    MOBILE_DEVICE = "Mobiel Toestel"
    SECURITY = "Security"
    APPLICATION_SUPPORT = "Applicatieondersteuning"
    OTHER = "Overig"


class ClassificationResult(BaseModel):
    categorie: Category
    te_valideren: bool
    security_flag: bool
    reden: str
