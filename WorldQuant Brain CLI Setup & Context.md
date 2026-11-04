# **Context for Gemini CLI: WorldQuant Brain Alpha Automation**

# **System Directive**

Your task as the CLI agent is to automate WorldQuant Brain alpha generation on this local machine. The end goal is to discover passing alphas (Sharpe \> 1.25, Fitness \>= 1.0) and submit them automatically to build a profitable portfolio for the user.

# **Phase 1: Environment & Client Setup**

You must use the `requests` library to handle WorldQuant's session-based HTTP Basic Authentication. Use this exact code pattern to authenticate and store the session cookies:

import requests
import os

session = requests.Session()
email = os.getenv('WQB_EMAIL')
password = os.getenv('WQB_PASSWORD')

response = session.post('https://api.worldquantbrain.com/authentication', auth=(email, password))
if response.status_code == 201:
    print("Authentication successful")

# **Phase 2: Alpha Generation Logic (Critical Context)**

The target simulation environment is **USA / D1 (1-Day Delay) / TOP3000**. In this environment, raw momentum-based price formulas fail. You must build mean-reversion alphas.

Failed Example to Avoid:

close / ts\_mean(close, 20\) \- 1Why it fails: It buys price spikes in a mean-reverting market, leading to a negative Sharpe ratio (e.g., \-0.87).

**Successful Blueprint to Iterate On:**

`rank(-1 * (close / ts_mean(close, 20) - 1))`*Why it works:*

* **Inversion (`-1 *`)**: Buys stocks trading below their moving average and shorts those above it.
* **Cross-Sectional Ranking (`rank()`)**: Forces a market-neutral distribution of long/short positions across the universe of 3000 stocks.

# **Phase 3: Simulation Parameters**

When constructing the API simulation payload for `POST https://api.worldquantbrain.com/simulations`, use these exact settings to ensure passing grades:

* `type`: "REGULAR"
* `settings`:
  * `instrumentType`: "EQUITY"
  * `region`: "USA"
  * `universe`: "TOP3000"
  * `delay`: 1
  * `decay`: 5 (CRUCIAL: Setting decay to 4 or 5 reduces turnover and pushes Fitness above 1.0)
  * `neutralization`: "SUBINDUSTRY"
  * `truncation`: 0.05
* `regular`: "\<THE\_ALPHA\_EXPRESSION\>"

# **Phase 4: Automation Loop**

Write a Python execution loop that:

* Iterates through different data fields (e.g., `close`, `volume`, `returns`, fundamental data) and timeframes (e.g., 5, 10, 20, 60 days) plugging them into the successful blueprint.
* Submits the alpha payload to the API.
* Polls the `GET https://api.worldquantbrain.com/simulations/{id}` endpoint with a delay.
* Evaluates the In-Sample (IS) Summary results. If **Sharpe \> 1.25** and **Fitness \>= 1.0**, log the expression to a success file and trigger the Out-of-Sample (OS) submission workflow.
