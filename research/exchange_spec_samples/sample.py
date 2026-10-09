import requests, time, json, statistics, datetime as dt
out=[]
def stats(bids,asks):
    bb,ba=bids[0][0],asks[0][0]; mid=(bb+ba)/2
    r={'spread_bps':(ba-bb)/mid*1e4,'mid':mid}
    for b in (1,5,10):
        lo,hi=mid*(1-b/1e4),mid*(1+b/1e4)
        r[f'bid_{b}']=sum(p*q for p,q in bids if p>=lo)
        r[f'ask_{b}']=sum(p*q for p,q in asks if p<=hi)
    r['max_depth_bps_bid']=(mid-bids[-1][0])/mid*1e4
    r['max_depth_bps_ask']=(asks[-1][0]-mid)/mid*1e4
    return r
for i in range(12):
    ts=dt.datetime.utcnow().isoformat()+'Z'
    for s in ('BTCUSDT','ETHUSDT'):
        try:
            j=requests.get('https://fapi.binance.com/fapi/v1/depth',params={'symbol':s,'limit':20},timeout=5).json()
            r=stats([(float(p),float(q)) for p,q in j['bids']],[(float(p),float(q)) for p,q in j['asks']])
            r.update(venue='binance',symbol=s,ts=ts,levels=20); out.append(r)
            j=requests.get('https://fapi.binance.com/fapi/v1/depth',params={'symbol':s,'limit':500},timeout=5).json()
            r=stats([(float(p),float(q)) for p,q in j['bids']],[(float(p),float(q)) for p,q in j['asks']])
            r.update(venue='binance',symbol=s,ts=ts,levels=500); out.append(r)
            j=requests.get('https://api.bybit.com/v5/market/orderbook',params={'category':'linear','symbol':s,'limit':200},timeout=5).json()['result']
            r=stats([(float(p),float(q)) for p,q in j['b']],[(float(p),float(q)) for p,q in j['a']])
            r.update(venue='bybit',symbol=s,ts=ts,levels=200); out.append(r)
        except Exception as e: print('err',s,e)
    time.sleep(5)
json.dump(out,open('ob_samples.json','w'),indent=1)
