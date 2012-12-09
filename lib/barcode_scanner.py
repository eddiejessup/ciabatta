rpc_key = '3a7a73eda4f48b1a3a1cb6b10a106d49a929af90'

def barcode_lookup(barcode, protocol='ean'):
    import xmlrpclib
    serv = xmlrpclib.ServerProxy('http://www.upcdatabase.com/xmlrpc')
    params = {'rpc_key': rpc_key}
    if protocol not in ['ean', 'upc']:
        raise Exception('Barcode protocol %s is not supported.' % protocol)
        
    params[protocol] = barcode
    result = serv.lookup(params)
    if result['status'] == 'fail':
        raise Exception(result['message'])
    return result
