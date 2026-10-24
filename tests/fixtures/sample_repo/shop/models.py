class Item:
    def __init__(self, price):
        self.price = price

class DiscountedItem(Item):
    def total(self):
        return self.price * 0.9
