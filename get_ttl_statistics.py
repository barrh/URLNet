import dns.resolver
from scipy import stats
from tld import get_tld



def get_ttl(domain_):
    try:
        soa_answer = dns.resolver.resolve(domain_, "SOA")
    except:
        return None
    return int(list(soa_answer.response.answer[-1])[0].to_text().split(' ')[-1])
ds = pd.read_parquet("data/Malicious_domain_detection/")
urls = ds["test"]["normalized_url"].to_list()
labels = ds["test"]["label"].to_list()
possible_init_strings = ["http://www.","https://www.","http://", "https://"]
malicious,benign = [],[]
given_is_ip = 0
given_is_subdomain = 0
is_subdomain = False
ttl_subdomain = []
ttl_orig_domain = []
for url,label in zip(urls,labels):
    try:
        tld = get_tld(url)
    except:
        given_is_ip+=1
        continue
    domain_ = url[:url.find(tld)] + tld

    for init_string in possible_init_strings:
        if domain_.startswith(init_string):
            domain_ = domain_[len(init_string):]
            break
    else:
        print(domain_)



    for init_string in possible_init_strings:
        if domain_.startswith(init_string):
            domain_ = domain_[len(init_string):]
            break
    domain_without_tld = domain_[:-len(tld)-1]
    is_subdomain = '.' in domain_without_tld
    if is_subdomain:
        two_tld = domain_without_tld[domain_without_tld.find('.')+1:]+"."+tld
    if is_subdomain and label:
        given_is_subdomain+=1
    ttl = get_ttl(domain_)
    if ttl==None:
        continue
    if is_subdomain:
        local_ttl = get_ttl(two_tld)
        ttl_subdomain.append(ttl)
        ttl_orig_domain.append(local_ttl)
    if label:
        malicious.append(ttl)
    else:
        benign.append(ttl)



tlds = [get_tld(url) for url in urls]
