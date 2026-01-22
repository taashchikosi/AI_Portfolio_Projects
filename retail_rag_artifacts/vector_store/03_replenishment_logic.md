# Replenishment Logic

If inventory_position < reorder_point:
recommended_order_qty = ceil(reorder_point − inventory_position)

This converts forecasts into standard buyer decisions.
