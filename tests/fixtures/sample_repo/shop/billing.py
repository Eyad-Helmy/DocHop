from shop.models import DiscountedItem
from . import models

def make_item(p):
    return DiscountedItem(p)

def checkout(p):
    item = make_item(p)
    return helper(item)

def helper(item):
    return item
def calculate_total(price, quantity):
    """
    Calculate the total price of products.

    Args:
        price: Price of one product.
        quantity: Number of products.

    Returns:
        Total price.
    """
    return price * quantity