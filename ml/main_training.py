import grpc
from grpc_client import HEClient

def run_test():
    client = HEClient()

    # テストデータ
    val1 = [1.0, 2.0]
    val2 = [3.0, 4.0]

    print(f"入力1: {val1}")
    print(f"入力2: {val2}")

    try:
        # 1. 暗号化
        ct1 = client.encrypt(val1)
        ct2 = client.encrypt(val2)
        print("✅ 暗号化成功")

        # 2. 加算のテスト ([1+3, 2+4] = [4.0, 6.0])
        ct_add = client.add(ct1, ct2)
        res_add = client.decrypt(ct_add)
        
        # MaxSlots分の配列が返ってくるため、入力長と同じ要素数だけスライスして確認
        print(f"✅ 加算結果: {[round(v, 4) for v in res_add[:2]]}")

        # 3. 乗算のテスト ([1*3, 2*4] = [3.0, 8.0])
        ct_mul = client.multiply(ct1, ct2)
        res_mul = client.decrypt(ct_mul)
        print(f"✅ 乗算結果: {[round(v, 4) for v in res_mul[:2]]}")

    except grpc.RpcError as e:
        print(f"gRPC通信エラー: {e.details()}")

if __name__ == "__main__":
    run_test()