# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: cycling-eta
#     language: python
#     name: python3
# ---

from gpx import read_gpx
import pandas as pd

df = read_gpx("data/200km_BRM.gpx")

df

df["time"] = pd.to_datetime(df["timestamp_ms"], unit="ms")

df.query("time > '2025-03-01 08:56:00'").head(20)
