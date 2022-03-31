import yfinance as yf
import datetime
import numpy as np
import mystic as my
import os


def main(market: str, stock_list: str):
    previous_date = datetime.datetime.today() - datetime.timedelta(weeks=52)
    end_date = datetime.datetime.today() - datetime.timedelta(days=1)

    raw_market_data = yf.download(market, start=previous_date, end=end_date, interval="1d", group_by='ticker')
    raw_stock_data = yf.download(stock_list, start=previous_date, end=end_date, interval="1d", group_by='ticker')

    market_data = generate_market_data(raw_market_data)
    parsed_data = generate_parsed_data(raw_stock_data, market_data, stock_list)

    portfolio = portfolio_optimiser(market_data, parsed_data, stock_list)
    print(portfolio[0])

    val_shares = list()
    budget = 50000

    for percent in portfolio[0]:
        val_shares.append(budget*percent)

    b = list()
    p = list()
    for i in parsed_data:
        b.append(parsed_data[i]['beta'])
        p.append(parsed_data[i]['value'][-1])

    obj = 0
    price = 0
    ret = 0
    num_shares = list()
    stock_split = stock_list.split(' ')
    x = portfolio[0]

    for j in range(len(x)):
        for k in range(len(x)):
            obj = obj + x[j]*x[k]*b[j]*b[k]
        num_shares.append(np.floor(val_shares[j]/p[j]))
        price += num_shares[j]*p[j]
        ret += portfolio[0][j]*parsed_data[stock_split[j]]['mean']

    for i in range(len(stock_split)):
        print(stock_split[i] + '; ' + str(num_shares[i]))
    print(price)
    print(ret)
    print(obj*market_data['var'])


def generate_market_data(data):
    print('Parsing market data')
    market_data = dict()
    market_data['value'] = data['Close'].values

    nan_list = list()
    s = 0
    for i in range(len(market_data['value'])):
        value = market_data['value'][i]

        if np.isnan(value):
            nan_list.append(i)
        else:
            s += value

    if len(nan_list) > 0:
        s = s/(len(market_data['value']) - len(nan_list))
        for i in nan_list:
            market_data['value'][i] = s

    market_data['returns'] = list()
    curr_value = market_data['value'][0]

    for i in range(1, len(market_data['value'])):
        r = (market_data['value'][i] - curr_value)/curr_value
        market_data['returns'].append(r)
        curr_value = market_data['value'][i]

    market_data['mean'] = np.mean(market_data['returns'])
    market_data['var'] = np.var(market_data['returns'])

    return market_data


def generate_parsed_data(stock_data, market_data, stock_list: str):
    parsed_data = dict()

    for ticker in stock_list.split(" "):
        print('Parsing data for ' + ticker)
        parsed_data[ticker] = dict()
        parsed_data[ticker]['value'] = stock_data[ticker]['Adj Close'].values

        nan_list = list()
        for i in range(len(market_data['value'])):
            value = parsed_data[ticker]['value'][i]

            if np.isnan(value):
                nan_list.append(i)
        if len(nan_list) == len(parsed_data[ticker]['value']):
            print('its fucking scuffed')

        if len(nan_list) > 0:
            for i in nan_list:
                j = 1
                while np.isnan(parsed_data[ticker]['value'][i + j]):
                    j += 1
                    if i + j > len(parsed_data[ticker]['value']) - 1:
                        j = -1
                        while np.isnan(parsed_data[ticker]['value'][i + j]):
                            if i + j < 0:
                                print('just go next')
                                exit(-10000)
                            j -= 1
                parsed_data[ticker]['value'][i] = parsed_data[ticker]['value'][i + j]

        parsed_data[ticker]['returns'] = list()

        curr_value = parsed_data[ticker]['value'][0]

        for i in range(1, len(parsed_data[ticker]['value'])):
            r = (parsed_data[ticker]['value'][i]-curr_value)/curr_value
            parsed_data[ticker]['returns'].append(r)
            curr_value = parsed_data[ticker]['value'][i]

        start_point = len(parsed_data[ticker]['returns']) - len(market_data['returns'])
        parsed_data[ticker]['mean'] = np.mean(parsed_data[ticker]['returns'])
        parsed_data[ticker]['var'] = np.var(parsed_data[ticker]['returns'])
        parsed_data[ticker]['beta'] = np.cov(parsed_data[ticker]['returns'][start_point:], market_data['returns'])[0][1]
        parsed_data[ticker]['beta'] = parsed_data[ticker]['beta']/np.var(market_data['returns'])

        if np.isnan(parsed_data[ticker]['beta']):
            parsed_data[ticker]['beta'] = 1
            print('We got a Nanner')
    print('Parsed stock data')
    return parsed_data


def portfolio_optimiser(market_data, stock_data, stock_list: str):
    print('Beginning portfolio generation')
    n = len(stock_list.split(' '))
    g = 200
    q = 10
    b = list()
    p = list()
    budget = 1
    for i in stock_data:
        b.append(stock_data[i]['beta'])
        p.append(stock_data[i]['value'][-1])

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

    @my.penalty.linear_inequality(weight_penalty, k=1e4)
    def penalty(x):
        return 0.0

    @my.constraints.normalized(mass=1)
    def constraints(x):
        return x

    return my.solvers.diffev2(objective, x0=x0, bounds=bounds, npop=n*q, penalty=penalty, constraint=constraints,
                              ftol=1e-8, gtol=g, disp=True, full_output=True, cross=.9, scale=.8, itermon=mon)


"""
def cov(list_1, list_2):
    mean_1 = np.mean(list_1)
    mean_2 = np.mean(list_2)

    sub_1 = [i - mean_1 for i in list_1]
    sub_2 = [i - mean_2 for i in list_2]

    numerator = sum([sub_1[i]*sub_2[i] for i in range(len(list_1))])
    denominator = len(list_1) - 1

    return numerator/denominator
"""


LARGE_CAP_STOCKHOLM = 'AAK.ST ABB.ST ADDT-B.ST AFRY.ST ALFA.ST ARION-SDB.ST ARJO-B.ST ASSA-B.ST AZN.ST ATCO-A.ST ' \
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


MARKET = '^OMX'

main(MARKET, LARGE_CAP_STOCKHOLM)
