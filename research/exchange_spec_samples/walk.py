import requests,time,json,datetime as dt,math
sizes=[1e3,5e3,1e4,2.5e4,5e4,1e5,2.5e5,5e5,1e6,2e6]
out=[]
def walk(levels,N,mid):
    rem=N;cost=0;qty=0
    for p,q in levels:
        v=p*q; take=min(v,rem); cost+=take; qty+=take/p; rem-=take
        if rem<=1e-9: break
    if rem>1e-9: return None
    vwap=cost/qty; return abs(vwap-mid)/mid*1e4
def one(venue,sym):
    if venue=='binance':
        j=requests.get('https://fapi.binance.com/fapi/v1/depth',params={'symbol':sym,'limit':1000},timeout=5).json(); b,a=j['bids'],j['asks']; T=j['E']
    else:
        j=requests.get('https://api.bybit.com/v5/market/orderbook',params={'category':'linear','symbol':sym,'limit':500},timeout=5).json()['result']; b,a=j['b'],j['a']; T=j['ts']
    b=[(float(p),float(q)) for p,q in b]; a=[(float(p),float(q)) for p,q in a]
    mid=(b[0][0]+a[0][0])/2
    r={'venue':venue,'symbol':sym,'exch_ts':T,'local_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'mid':mid,'spread_bps':(a[0][0]-b[0][0])/mid*1e4,'levels':len(b)}
    for x in (1,5,10,25):
        r[f'bid_{x}bps']=sum(p*q for p,q in b if p>=mid*(1-x/1e4)); r[f'ask_{x}bps']=sum(p*q for p,q in a if p<=mid*(1+x/1e4))
    r['book_reach_bps']=min((mid-b[-1][0])/mid,(a[-1][0]-mid)/mid)*1e4
    r['buy_slip']={int(N):walk(a,N,mid) for N in sizes}; r['sell_slip']={int(N):walk(b,N,mid) for N in sizes}
    return r
for i in range(10):
    for v in ('binance','bybit'):
        for s in ('BTCUSDT','ETHUSDT'):
            try: out.append(one(v,s))
            except Exception as e: print('err',v,s,e)
    time.sleep(6)
json.dump(out,open('walk_samples.json','w'),indent=1)
print('done',len(out))
