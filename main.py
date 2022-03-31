from sklearn import linear_model
import yfinance as yf
import datetime
import numpy as np
from pandas import DataFrame
import mystic as my


def parse_data(data):
    """Fixes any NaN portions of the data and parses it into a nice list"""
    nan_list = list()
    fix = None

    for i in range(len(data)):
        if np.isnan(data[i]) & (fix is None):
            nan_list.append(i)
        else:
            if np.isnan(data[i]) & (fix is not None):
                data[i] = fix
            else:
                data[i] = int(data[i])
                fix = data[i]
                if len(nan_list) > 0:
                    for index in nan_list:
                        data[index] = fix
                    nan_list.clear()
    return data


def orthogonalize(base: DataFrame, projected: DataFrame, regress: linear_model.LinearRegression):
    """
    Orthogonalizes potentially correlated indexes using a Gram-Schmidt-esque process
    :param base: the base index
    :param projected: the index to be orthogonalised
    :param regress: the regression model of the projected onto the base
    :return: the orthogonalized index
    """
    try:
        regress.fit(base, projected)
    except ValueError:
        regress.fit(base.reshape(-1, 1), projected)
    else:
        print('Encountered an error while trying to regress in the orthogonalization which could no be resolved by '
              'reshaping')
        exit(-1)

    return np.add(projected, -np.add(regress.intercept_, regress.coef_*base))


def portfolio_optimiser(betas, stock_list: str):
    """
    A portfolio optimizer using the minimum variance approach
    :param betas: covariance between stocks and markets
    :param stock_list: the names of the stocks in question
    :return: an optimized vector for the weights of the portfolio
    """
    print('Beginning portfolio generation')
    n = len(stock_list.split(' '))
    g = 200
    q = 10
    b = list()
    p = list()
    budget = 1

    for stock in stock_list.split(' '):
        b.append(betas[stock][0])

    bounds = [(0, 1)]*n
    x0 = [1/n]*n
    mon = my.monitors.VerboseMonitor(10)

    def objective(x):
        obj = 0

        for j in range(len(x)):
            for k in range(len(x)):
                obj = obj + x[j]*x[k]*b[j]*b[k]

        return obj

    def weight_penalty(x):
        return np.sum(x) - 1

    def budget_penalty(x):
        pen = 0
        for j in range(len(x)):
            pen = pen + x[j]*p[j]
        return pen - budget

    def budget_lower(x):
        pen = 0
        for j in range(len(x)):
            pen = pen + x[j]*p[j]
        return 0.8*budget - pen

    @my.penalty.linear_equality(weight_penalty, k=1e4)
    def penalty(x):
        return 0.0

    @my.constraints.normalized(mass=1)
    def constraints(x):
        return x

    return my.solvers.diffev2(objective, x0=x0, bounds=bounds, npop=n*q, penalty=penalty, constraint=constraints,
                              ftol=1e-8, gtol=g, disp=True, full_output=True, cross=.9, scale=.8, itermon=mon)


reg = linear_model.LinearRegression()

start_date = datetime.datetime.today() - datetime.timedelta(weeks=1)
end_date = datetime.datetime.today() - datetime.timedelta(days=1)

MARKET_TICKERS = '^OMX SPY'

'''
STOCK_TICKERS = 'AAK.ST ABB.ST ADDT-B.ST AFRY.ST ALFA.ST ARION-SDB.ST ARJO-B.ST ASSA-B.ST AZN.ST ATCO-A.ST ' \
                'ATCO-B.ST ATRLJ-B.ST ALIV-SDB.ST AZA.ST AXFO.ST BEIJ-B.ST BETS-B.ST BHG.ST BILL.ST BOL.ST ' \
                'BRAV.ST BURE.ST CAST.ST CATE.ST CINT.ST CORE-A.ST CORE-B.ST CORE-D.ST CORE-PREF.ST ' \
                'DOM.ST ELUX-A.ST ELUX-B.ST EPRO-B.ST EKTA-B.ST EPI-A.ST EPI-B.ST EQT.ST ERIC-A.ST ERIC-B.ST ' \
                'ESSITY-A.ST ESSITY-B.ST EVO.ST FABG.ST BALD-B.ST FPAR-A.ST FPAR-D.ST FPAR-PREF.ST FOI-B.ST ' \
                'GETI-B.ST SHB-A.ST SHB-B.ST HEM.ST HM-B.ST HEXA-B.ST HPOL-B.ST HOLM-A.ST HOLM-B.ST HUFV-A.ST ' \
                'HUSQ-A.ST HUSQ-B.ST ICA.ST INDU-A.ST INDU-C.ST INDT.ST INTRUM.ST INVE-A.ST INVE-B.ST JM.ST ' \
                'KIND-SDB.ST KINV-A.ST KINV-B.ST KLED.ST LATO-B.ST LIFCO-B.ST LOOMIS.ST LUND-B.ST LUNE.ST ' \
                'LUMI.ST MCOV-B.ST TIGO-SDB.ST MYCR.ST NCC-A.ST NCC-B.ST NIBE-B.ST NOBI.ST NOLA-B.ST NDA-SE.ST ' \
                'NENT-A.ST NENT-B.ST SAVE.ST NYF.ST PNDX-B.ST PEAB-B.ST PLAZ-B.ST RATO-A.ST RATO-B.ST ' \
                'RESURS.ST SAAB-B.ST SAGA-A.ST SAGA-B.ST SAGA-D.ST SBB-B.ST SBB-D.ST SAND.ST SCA-A.ST SCA-B.ST ' \
                'SDIP-B.ST SDIP-PREF.ST SEB-A.ST SEB-C.ST SECT-B.ST SECU-B.ST SINCH.ST SKA-B.ST SKF-A.ST ' \
                'SKF-B.ST SSAB-A.ST SSAB-B.ST SF.ST STE-A.ST STE-R.ST STOR-B.ST SWEC-A.ST SWEC-B.ST SWED-A.ST ' \
                'SWMA.ST SOBI.ST TEL2-A.ST TEL2-B.ST TELIA.ST THULE.ST TIETOS.ST 8TRA.ST TREL-B.ST TRUE-B.ST ' \
                'VNE-SDB.ST VITR.ST VOLV-A.ST VOLV-B.ST VOLCAR-B.ST WALL-B.ST WIHL.ST'
'''

STOCK_TICKERS = 'AAK.ST ADDT-B.ST'

MARKET_TICKERS_SPLIT = MARKET_TICKERS.split(' ')
STOCK_TICKERS_SPLIT = STOCK_TICKERS.split(' ')

market_data = yf.download(MARKET_TICKERS, start=start_date, end=end_date, interval='1d', group_by='ticker')
stock_data = yf.download(STOCK_TICKERS, start=start_date, end=end_date, interval='1d', group_by='ticker')

parsed_market_data = dict()

parsed_stock_data = dict()
stock_betas = dict()
stock_alphas = dict()


for ticker in MARKET_TICKERS_SPLIT:
    parsed_market_data[ticker] = dict()

    if len(MARKET_TICKERS_SPLIT) > 1:
        parsed_market_data[ticker]['value'] = market_data[ticker]['Adj Close']
        parsed_market_data[ticker]['value'] = parse_data(parsed_market_data[ticker]['value'])

    else:
        parsed_market_data[ticker]['value'] = market_data['Adj Close'].values

    parsed_market_data[ticker]['returns'] = list()
    curr_value = parsed_market_data[ticker]['value'][0]

    for i in range(1, len(parsed_market_data[ticker]['value'])):
        parsed_market_data[ticker]['returns'].append((parsed_market_data[ticker]['value'][i] - curr_value)/curr_value)
        curr_value = parsed_market_data[ticker]['value'][i]
    parsed_market_data[ticker]['returns'] = np.array(parsed_market_data[ticker]['returns'])


if len(MARKET_TICKERS_SPLIT) > 1:
    for i in range(len(MARKET_TICKERS_SPLIT)):
        for j in range(i+1, len(MARKET_TICKERS_SPLIT[i:])):
            reduce = MARKET_TICKERS_SPLIT[j]
            reducer = MARKET_TICKERS_SPLIT[i]

            parsed_market_data[reduce]['returns'] = orthogonalize(parsed_market_data[reducer]['returns'],
                                                                  parsed_market_data[reduce]['returns'], reg)


market_returns = DataFrame()

for ticker in MARKET_TICKERS_SPLIT:
    if len(MARKET_TICKERS_SPLIT) > 1:
        market_returns[ticker] = parsed_market_data[ticker]['returns']
    else:
        market_returns = parsed_market_data[ticker]['returns'].reshape(-1, 1)

for ticker in STOCK_TICKERS_SPLIT:
    parsed_stock_data[ticker] = dict()

    parsed_stock_data[ticker]['value'] = stock_data[ticker]['Adj Close'].values
    parsed_stock_data[ticker]['value'] = parse_data(parsed_stock_data[ticker]['value'])

    parsed_stock_data[ticker]['returns'] = list()
    curr_value = parsed_stock_data[ticker]['value'][0]

    for i in range(1, len(parsed_stock_data[ticker]['value'])):
        parsed_stock_data[ticker]['returns'].append((parsed_stock_data[ticker]['value'][i] - curr_value)/curr_value)
        curr_value = parsed_stock_data[ticker]['value'][i]

    reg.fit(market_returns, parsed_stock_data[ticker]['returns'][-len(parsed_market_data['^OMX']['returns']):])
    stock_betas[ticker] = reg.coef_
    stock_alphas[ticker] = reg.intercept_

portfolio = portfolio_optimiser(stock_betas, STOCK_TICKERS)
print(portfolio[0])

val_shares = list()
budget = 50000

for percent in portfolio[0]:
    val_shares.append(budget*percent)

b = list()
p = list()
for i in STOCK_TICKERS_SPLIT:
    b.append(stock_betas[i][0])
    p.append(parsed_stock_data[i]['value'][-1])

obj = 0
price = 0
ret = 0
num_shares = list()

x = portfolio[0]

for j in range(len(x)):
    for k in range(len(x)):
        obj = obj + x[j]*x[k]*b[j]*b[k]
    num_shares.append(np.floor(val_shares[j]/p[j]))
    price += num_shares[j]*p[j]
    ticker = STOCK_TICKERS_SPLIT[j]
    ret += portfolio[0][j]*np.mean(parsed_stock_data[ticker]['returns'])

for i in range(len(STOCK_TICKERS_SPLIT)):
    print(STOCK_TICKERS_SPLIT[i] + '; ' + str(num_shares[i]))
print(price)
print(ret)
print(obj*np.var(parsed_market_data['^OMX']['returns']))
