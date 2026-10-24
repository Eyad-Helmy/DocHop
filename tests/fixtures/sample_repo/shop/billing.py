from shop.models import DiscountedItem
from . import models

def make_item(p):
    return DiscountedItem(p)

def checkout(p):
    item = make_item(p)
    return helper(item)

def helper(item):
    return item
