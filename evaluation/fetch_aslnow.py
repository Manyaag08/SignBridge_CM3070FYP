"""Download the ASLNow! landmark dataset (sid220/asl-now-fingerspelling, MIT licence, ~8 MB) into a folder.
Only missing files are fetched; HTTP 429 responses are waited out using Retry-After. Usage: fetch_aslnow.py OUT_DIR"""
import concurrent.futures as cf, json, os, sys, threading, time, urllib.error, urllib.request

REPO = "sid220/asl-now-fingerspelling"
OUT = sys.argv[1]
names = [f["rfilename"] for f in json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}", timeout=60))["siblings"]
         if f["rfilename"].endswith(".json")]
todo = [n for n in names if not os.path.exists(os.path.join(OUT, n))]
print(f"ASLNow!: {len(names)} files listed, {len(todo)} to download", flush=True)
lock = threading.Lock(); done = 0

def get(n):
    global done
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{n}"
    for attempt in range(30):
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
            json.loads(data)                                   # keep only complete, valid files
            os.makedirs(os.path.join(OUT, os.path.dirname(n)), exist_ok=True)
            open(os.path.join(OUT, n), "wb").write(data)
            with lock:
                done += 1
                if done % 250 == 0: print(f"  {done}/{len(todo)}", flush=True)
            return
        except urllib.error.HTTPError as e:
            time.sleep(int(e.headers.get("Retry-After") or 0) or min(300, 15 * (attempt + 1)))
        except Exception:
            time.sleep(5)

with cf.ThreadPoolExecutor(6) as ex:          # 6 workers x 2 requests per file stays under 3000 requests / 5 min
    list(ex.map(get, todo))
present = sum(os.path.exists(os.path.join(OUT, n)) for n in names)
print(f"ASLNow!: {present}/{len(names)} files present")
sys.exit(0 if present == len(names) else 1)
