from swtipy import hello

def test_hello():
    result = hello()   # 呼叫函式
    assert result == "hello from python, testing package"