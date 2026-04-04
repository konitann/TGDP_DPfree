import sys
import os
import grpc
import he_service_pb2
import he_service_pb2_grpc

class HEClient:
    def __init__(self, host='localhost', port=50051):
        # 最大メッセージサイズを50MBに設定
        MAX_MESSAGE_LENGTH = 50 * 1024 * 1024
        options = [
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ]
        # options引数を追加
        self.channel = grpc.insecure_channel(f'{host}:{port}', options=options)
        self.stub = he_service_pb2_grpc.HEServiceStub(self.channel)

    def encrypt(self, values):
        req = he_service_pb2.EncryptRequest(plain_values=values)
        res = self.stub.Encrypt(req)
        return res.ciphertext

    def decrypt(self, ciphertext):
        req = he_service_pb2.DecryptRequest(ciphertext=ciphertext)
        res = self.stub.Decrypt(req)
        # protobufの定義に従い plain_values を返す
        return res.plain_values
    
    def compute_gradient(self, encrypted_weights, x_batch, y_batch, n_samples, n_features, lambda_reg):
        """
        暗号化された重みと平文データを送り、サーバー側で計算された暗号化勾配を取得する
        """
        request = he_service_pb2.GradientRequest(
            encrypted_weights=encrypted_weights,
            x_batch=x_batch,
            y_batch=y_batch,
            n_samples=n_samples,
            n_features=n_features,
            lambda_reg=lambda_reg
        )
        response = self.stub.ComputeGradient(request)
        return response.encrypted_gradient

    def add(self, ciphertext1, ciphertext2):
        req = he_service_pb2.AddRequest(ciphertext1=ciphertext1, ciphertext2=ciphertext2)
        res = self.stub.Add(req)
        return res.result_ciphertext

    def multiply(self, ciphertext1, ciphertext2):
        req = he_service_pb2.MultiplyRequest(ciphertext1=ciphertext1, ciphertext2=ciphertext2)
        res = self.stub.Multiply(req)
        return res.result_ciphertext