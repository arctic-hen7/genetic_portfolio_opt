#!/usr/bin/env python
# Runs the portfolio optimiser on Modal.

import modal

app = modal.App("portfolio-optimiser")

@app.function(cpu=64)
