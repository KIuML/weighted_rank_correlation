product = dict(
    norm=lambda a, b: a * b,
    conorm=lambda a, b: a + b - a * b
)

# Łukasiewicz t-Conorm
luka = dict(
    norm=lambda a, b: max(a + b - 1, 0),
    conorm=lambda a, b: min(a + b, 1)
)