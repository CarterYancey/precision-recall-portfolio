# Nasdaq Data Link Python Client

## Configuration

| Option | Explanation | Example |
|---|---|---|
| api_key | Your access key | `tEsTkEy123456789` | Used to identify who you are and provide full access. |
| use_retries | Whether API calls which return statuses in `retry_status_codes` should be automatically retried | True
| number_of_retries | Maximum number of retries that should be attempted. Only used if `use_retries` is True | 5
| max_wait_between_retries | Maximum amount of time in seconds that should be waited before attempting a retry. Only used if `use_retries` is True | 8
| retry_backoff_factor | Determines the amount of time in seconds that should be waited before attempting another retry. Note that this factor is exponential so a `retry_backoff_factor` of 0.5 will cause waits of [0.5, 1, 2, 4, etc]. Only used if `use_retries` is True | 0.5
| retry_status_codes | A list of HTTP status codes which will trigger a retry to occur. Only used if `use_retries` is True| [429, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511]

### Local API Key file

The default configuration file location is `~/.nasdaq/data_link_apikey`.  The
client will attempt to load this file if it exists.  Note: if the file exists
and empty, a ValueError will be thrown.

## TABLES

Let's get started with a few basic examples of how to access the data using the various tools that are available to you: the Web API, Python, R, and Excel. These simple examples can be combined to produce more complex queries.

There are 5 tables associated with this database. You can see these tables visually presented here. The primary table is
SHARADAR/SEP
. The other tables are
SHARADAR/ACTIONS
,
SHARADAR/TICKERS
,
SHARADAR/METRICS
and
SHARADAR/INDICATORS
.

In each table data are presented in tabular format with indicators represented in columns.

When you query data from a particular table you can specify "filters" which limit the data that is retrieved. If you do not specify a filter the entire table, all rows and columns, will be returned but limited to 10,000 rows.

There are 3 filter columns in the primary
SHARADAR/SEP
table that you can use to specify which data to retrieve, these are:
ticker
,
date
and
lastupdated
.

When you integrate this API you should not assume a particular column order as additional indicators will be added in the future.

Data is not sorted by default and you should sort the data upon retrieval.

## EXAMPLES

In these examples we will assume that you have the associated package installed where necessary. In the case of the Web API, no Nasdaq Data Link package needs to be installed. You can find installation instructions and generic documentation for each of the tools here. The data retrieval methods described here are referred to under the "tables" section of Nasdaq Data Link's generic documentation, rather than the "time-series" section.

```
>>> # To retrieve all data for a specified ticker:
>>> 
>>> nasdaqdatalink.get_table('SHARADAR/SEP', ticker='AAPL')
     ticker       date     open     high      low    close        volume  closeadj  closeunadj lastupdated
None                                                                                                      
0      AAPL 2026-07-10  314.720  316.910  312.170  315.320  3.246600e+07   315.320      315.32  2026-07-11
1      AAPL 2026-07-09  310.445  316.530  308.160  316.220  4.488200e+07   316.220      316.22  2026-07-09
2      AAPL 2026-07-08  311.660  314.810  307.051  313.390  3.846600e+07   313.390      313.39  2026-07-08
3      AAPL 2026-07-07  315.290  315.480  310.150  310.660  4.209700e+07   310.660      310.66  2026-07-07
4      AAPL 2026-07-06  307.680  314.200  307.010  312.660  4.913000e+07   312.660      312.66  2026-07-06
...     ...        ...      ...      ...      ...      ...           ...       ...         ...         ...
7169   AAPL 1998-01-07    0.168    0.170    0.154    0.156  1.041622e+09     0.131       17.50  2026-05-11
7170   AAPL 1998-01-06    0.142    0.178    0.132    0.169  1.812474e+09     0.142       18.94  2026-05-11
7171   AAPL 1998-01-05    0.147    0.148    0.136    0.142  6.518740e+08     0.119       15.88  2026-05-11
7172   AAPL 1998-01-02    0.122    0.145    0.120    0.145  7.181100e+08     0.122       16.25  2026-05-11
7173   AAPL 1997-12-31    0.117    0.122    0.116    0.117  4.063580e+08     0.098       13.13  2026-05-11

[7174 rows x 10 columns]
>>> 
>>> # To retrieve data for multiple specified tickers:
>>> 
>>> nasdaqdatalink.get_table('SHARADAR/SEP', ticker=['AAPL','TSLA'])
/home/carter/precision-recall-portfolio/.venv/lib/python3.12/site-packages/nasdaqdatalink/get_table.py:38: UserWarning: To request more pages, please set paginate=True in your         nasdaqdatalink.get_table() call. For more information see our documentation:         https://github.com/Nasdaq/data-link-python/blob/main/FOR_ANALYSTS.md#things-to-note
  warnings.warn(Message.WARN_PAGE_LIMIT_EXCEEDED, UserWarning)
     ticker       date     open     high      low    close       volume  closeadj  closeunadj lastupdated
None                                                                                                     
0      TSLA 2026-07-10  410.490  413.160  402.810  407.760   32768000.0   407.760      407.76  2026-07-11
1      TSLA 2026-07-09  393.938  407.860  390.863  406.550   36520000.0   406.550      406.55  2026-07-09
2      TSLA 2026-07-08  399.375  399.630  390.510  394.060   33242000.0   394.060      394.06  2026-07-08
3      TSLA 2026-07-07  416.970  419.550  401.880  402.900   37963000.0   402.900      402.90  2026-07-07
4      TSLA 2026-07-06  397.320  420.000  390.500  419.770   51649000.0   419.770      419.77  2026-07-06
...     ...        ...      ...      ...      ...      ...          ...       ...         ...         ...
9995   AAPL 2002-10-24    0.268    0.272    0.260    0.262  174748000.0     0.220       14.69  2026-05-11
9996   AAPL 2002-10-23    0.261    0.268    0.259    0.266  209037000.0     0.223       14.88  2026-05-11
9997   AAPL 2002-10-22    0.259    0.266    0.255    0.263  218148000.0     0.220       14.70  2026-05-11
9998   AAPL 2002-10-21    0.255    0.261    0.250    0.260  238521000.0     0.218       14.56  2026-05-11
9999   AAPL 2002-10-18    0.250    0.256    0.249    0.256  288299000.0     0.215       14.34  2026-05-11

[10000 rows x 10 columns]
>>> 
>>> # To retrieve data for a specified date:
>>> 
>>> nasdaqdatalink.get_table('SHARADAR/SEP', date='2017-10-30')
     ticker       date   open   high     low   close       volume  closeadj  closeunadj lastupdated
None                                                                                               
0      ZYNE 2017-10-30   9.80  10.27   9.630   9.760   256138.000     9.760       9.760  2018-06-13
1      ZYME 2017-10-30   8.32   8.45   8.270   8.390     3313.000     8.390       8.390  2018-06-13
2     ZXAIY 2017-10-30   1.21   1.21   1.199   1.200     4265.000     1.200       1.200  2019-04-26
3       ZWS 2017-10-30  25.52  25.68  25.270  25.460   798000.000    11.898      25.460  2026-05-20
4      ZVRA 2017-10-30  58.40  60.80  56.800  60.000     2293.688    60.000       3.750  2023-03-01
...     ...        ...    ...    ...     ...     ...          ...       ...         ...         ...
5701   AACH 2017-10-30   8.55   8.55   8.110   8.200    68413.000     8.200       8.200  2020-04-15
5702   AACG 2017-10-30   4.70   4.70   4.638   4.638     1749.000     1.861       4.638  2019-10-22
5703  AAAP1 2017-10-30  80.35  80.70  80.000  80.500  8853000.000    80.500      80.500  2026-06-23
5704     AA 2017-10-30  47.75  48.38  47.370  47.420  2072000.000    45.174      47.420  2026-05-19
5705      A 2017-10-30  67.80  67.93  67.100  67.490   847000.000    63.182      67.490  2026-06-30

[5706 rows x 10 columns]
>>> 
>>> # To retrieve data for a specified data range:
>>> 
>>> nasdaqdatalink.get_table('SHARADAR/SEP', date={'gte':'2017-01-01', 'lte':'2017-10-30'}, ticker='AAPL')
     ticker       date    open    high     low   close       volume  closeadj  closeunadj lastupdated
None                                                                                                 
0      AAPL 2017-10-30  40.972  42.017  40.930  41.680  178803000.0    38.834      166.72  2026-05-11
1      AAPL 2017-10-27  39.822  40.900  39.675  40.763  177817000.0    37.979      163.05  2026-05-11
2      AAPL 2017-10-26  39.307  39.457  39.195  39.352   68002000.0    36.665      157.41  2026-05-11
3      AAPL 2017-10-25  39.227  39.388  38.818  39.102   84828000.0    36.432      156.41  2026-05-11
4      AAPL 2017-10-24  39.072  39.355  39.050  39.275   71029000.0    36.593      157.10  2026-05-11
...     ...        ...     ...     ...     ...     ...          ...       ...         ...         ...
204    AAPL 2017-01-09  29.488  29.858  29.485  29.747  134248000.0    27.374      118.99  2026-05-11
205    AAPL 2017-01-06  29.195  29.540  29.117  29.477  127008000.0    27.125      117.91  2026-05-11
206    AAPL 2017-01-05  28.980  29.216  28.953  29.152   88774000.0    26.826      116.61  2026-05-11
207    AAPL 2017-01-04  28.962  29.128  28.938  29.005   84472000.0    26.691      116.02  2026-05-11
208    AAPL 2017-01-03  28.950  29.082  28.690  29.038  115127000.0    26.720      116.15  2026-05-11

[209 rows x 10 columns]
>>> 
>>> # API calls are automatically truncated at 10,000 rows, which can be circumvented as follows:
>>> 
>>> nasdaqdatalink.get_table('SHARADAR/SEP', date={'gte':'2017-10-01','lte':'2017-11-03'}, paginate=True)
       ticker       date   open    high    low  close     volume  closeadj  closeunadj lastupdated
None                                                                                              
0        ZYNE 2017-11-03   9.83  10.500   9.83  10.33   531495.0    10.330       10.33  2020-05-01
1        ZYNE 2017-11-02   9.59  10.200   9.51   9.85   323496.0     9.850        9.85  2020-05-01
2        ZYNE 2017-11-01   9.70   9.930   9.41   9.69   301604.0     9.690        9.69  2018-06-13
3        ZYNE 2017-10-31   9.74   9.949   9.63   9.80   232742.0     9.800        9.80  2018-06-13
4        ZYNE 2017-10-30   9.80  10.270   9.63   9.76   256138.0     9.760        9.76  2018-06-13
...       ...        ...    ...     ...    ...    ...        ...       ...         ...         ...
142418      A 2017-10-06  65.68  66.380  65.66  66.36  1405000.0    62.124       66.36  2026-06-30
142419      A 2017-10-05  65.85  65.990  65.52  65.70  1974000.0    61.506       65.70  2026-06-30
142420      A 2017-10-04  65.15  65.870  65.15  65.83   746000.0    61.628       65.83  2026-06-30
142421      A 2017-10-03  65.06  65.780  65.06  65.15  1217000.0    60.991       65.15  2026-06-30
142422      A 2017-10-02  64.29  65.070  64.21  64.87  1694000.0    60.729       64.87  2026-06-30

[142423 rows x 10 columns]
>>> 
>>> # TICKER METADATA
>>> nasdaqdatalink.get_table('SHARADAR/TICKERS', table='SEP', ticker='AAPL')
     table  permaticker ticker       name exchange isdelisted               category     cusips  ...  lastupdated firstadded firstpricedate lastpricedate firstquarter lastquarter                                         secfilings           companysite
None                                                                                             ...                                                                                                                                                       
0      SEP       199059   AAPL  APPLE INC   NASDAQ          N  Domestic Common Stock  037833100  ...   2026-07-11 2014-09-24     1986-01-01    2026-07-10   1992-12-31  2026-03-31  https://www.sec.gov/cgi-bin/browse-edgar?actio...  http://www.apple.com

[1 rows x 28 columns]
>>> 
>>> # INDICATOR DESCRIPTIONS BY TABLE
>>> nasdaqdatalink.get_table('SHARADAR/INDICATORS', table='SEP')
     table    indicator isfilter isprimarykey                                              title                                        description           unittype
None                                                                                                                                                                  
0      SEP       volume        N            N                            Volume - Split Adjusted  The daily traded volume across all exchanges; ...            numeric
1      SEP       ticker        Y            Y                                      Ticker Symbol  The ticker is a unique identifier for a securi...               text
2      SEP         open        N            N                        Open Price - Split Adjusted  The official exchange opening price; adjusted ...          USD/share
3      SEP          low        N            N                         Low Price - Split Adjusted  The low share price; adjusted for stock splits...          USD/share
4      SEP  lastupdated        Y            N                                  Last Updated Date  The last date at which this line item was upda...  date (YYYY-MM-DD)
5      SEP         high        N            N                        High Price - Split Adjusted  The high share price; adjusted for stock split...          USD/share
6      SEP         date        Y            Y                                         Price Date          The trade date of the price observations.  date (YYYY-MM-DD)
7      SEP   closeunadj        N            N                           Close Price - Unadjusted  The official exchange close price; not adjuste...          USD/share
8      SEP     closeadj        N            N  Close Price - Adjusted for Splits Dividends an...  The official exchange close price; adjusted fo...          USD/share
9      SEP        close        N            N                       Close Price - Split Adjusted  The official exchange close price; adjusted fo...          USD/share
>>> 
>>> # CORPORATE ACTIONS BY TICKER
>>> nasdaqdatalink.get_table('SHARADAR/ACTIONS', ticker='AAPL')
           date         action ticker       name      value contraticker     contraname
None                                                                                   
0    2026-05-11       dividend   AAPL  APPLE INC    0.27000          N/A            N/A
1    2026-02-09       dividend   AAPL  APPLE INC    0.26000          N/A            N/A
2    2025-11-10       dividend   AAPL  APPLE INC    0.26000          N/A            N/A
3    2025-08-11       dividend   AAPL  APPLE INC    0.26000          N/A            N/A
4    2025-05-12       dividend   AAPL  APPLE INC    0.26000          N/A            N/A
...         ...            ...    ...        ...        ...          ...            ...
57   2012-10-04  acquisitionof   AAPL  APPLE INC  358.70000         AUTH  AUTHENTEC INC
58   2012-08-09       dividend   AAPL  APPLE INC    0.09464          N/A            N/A
59   2005-02-28          split   AAPL  APPLE INC    2.00000          N/A            N/A
60   2000-06-21          split   AAPL  APPLE INC    2.00000          N/A            N/A
61   1997-12-31      initiated   AAPL  APPLE INC        NaN         AAPL            N/A

[62 rows x 7 columns]
```

---

# Knowledge Base

## Adjustment Overview

The indicator "closedadj" is backward-adjusted. This means today's adjusted price will always equal the price traded in the market. Adjustments apply only to historical data; the further back you go into the past, the greater the cumulative adjustments are likely to be, as corporate actions accumulate. See the Notes section for more information on adjustments.

The following sections describe the adjustments made for various corporate actions.

## Cash Dividends
Adjustment Ratio = (Close Price + Dividend Amount) / (Close Price)

Example: AAPL had a dividend of $0.47 on 2014-08-07. The close price on that day was $94.48. The unadjusted close price for the previous day was $94.96. The adjusted historical prices are calculated like this:

dividend_amount = 0.47

current_close = 94.48

adjustment_ratio
  = (current_close + dividend_amount) / current_close
  = (94.48 + 0.47) / 94.48
  = 1.0049745977984759

previous_close = 94.96

adjusted_previous_close
  = previous_close / adjustment_ratio
  = 94.96 / 1.0049745977984759
  = 94.490
and the same adjustment propagates all through the stock's history.

## Stock Dividends

Adjustment Ratio = (New Float) / (Old Float)

Example: BIOL had a 0.5% stock dividend on 2014-03-12. Shareholders were given 1 new share per 200 shares of BIOL already held.

new_float = 1.005 * old_float  by definition

adjustment_ratio
  = new_float / old_float
  = 1.005

current_close = 2.9

previous_close = 2.83

adjusted_previous_close
  = previous_close / adjustment_ratio
  = 2.83 / 1.005
  = 2.8159
  
## Splits

Adjustment Ratio = (New Float) / (Old Float)

For example, CPK had a 3 for 2 split on 2014-09-09. Shareholders were given 3 shares per 2 shares previously held of CPK.

2 * new_float = 3 * old_float

adjustment_ratio
  = new_float / old_float
  = 3/2
  = 1.5

current_close = 45.11

previous_close = 69.41

adjusted_previous_close
  = previous_close / adjustment_ratio
  = 69.41 / 1.5
  = 46.273
Spinoffs
For spinoffs, we assume that you sell the spin-off stock at its open price, and use the proceeds to buy back the parent stock at its open price. This is the methodology used by most stock data providers.

A few stock data providers assume that you sell the spin-off stock at the close price, and buy back the parent stock at its close price. The difference between the two methods is usually minimal but it's non-zero.

Adjustment Ratio = 1 + (Spinoff Open Price * Spinoff Shares) / (Parent Open Price * Parent Shares)

For example, ADP spun off CDK on 2014-10-01. ADP shareholders were given 1 share of CDK for every 3 shares of ADP they held.

parent_open_price = 73.03   <-- ADP

spinoff_open_price = 30.13  <-- CDK

spinoff_shares / parent_shares = 1 / 3  per the terms of the spinoff

adjustment_ratio
  = 1 + (spinoff_open_price / parent_open_price) * (spinoff_shares / parent_shares)
  = 1 + (30.13 / 73.03) * (1 / 3)
  = 1 + 0.13752
  = 1.13752

parent_previous_close = 83.08

parent_adjusted_previous_close
  = parent_previous_close / adjustment_ratio
  = 83.08 / 1.13752
  = 73.036
