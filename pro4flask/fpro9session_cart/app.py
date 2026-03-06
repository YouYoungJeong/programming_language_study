from flask import Flask, render_template, request, session, make_response, redirect, url_for
from datetime import timedelta 

app = Flask(__name__)
# secret_key 설정
app.secret_key = "abcdef123456"
# 세션 만료 5분
app.permanent_session_lifetime = timedelta(minutes=5)

# 상품 만들기list(dict{}) - DB사용전이라
products=[
    {"id":1,"name":"노트북","price":3500000},
    {"id":2,"name":"키보드","price":50000},
    {"id":3,"name":"마우스","price":35000},
    {"id":4,"name":"모니터","price":1500000},
]

@app.route("/")
def product_list():
    return render_template("products.html", products=products)

@app.route("/cart")
def show_cart():
    # 세션에서 카트를 꺼내, 없으면 빈바구니를 꺼내 - 카트가 기억하는거야
    py_cart = session.get("se_cart", {})
    total = sum(info["price"] * info["qty"] for info in py_cart.values())
    ht_cart = py_cart
    return render_template("cart.html", ht_cart=ht_cart, total=total)

@app.route("/add/<int:product_id>")# 받아오는 값이름은 내맘대로 써도 돼
def add_to_cart(product_id):
    # print(product_id)
    # 세션 cart가 없으면 빈 dict로 생성
    # 세션에서 읽어온 카트내용을 cart변수에 저장
    py_cart = session.get("se_cart", {})

    # next(..., None) : 묶음형 자료에서 다음 값 1개를 꺼내는 함수
    # 주문 상품이 최종적으로 product에 기억됨
    product = next((p for p in products if p["id"] == product_id), None)

    # 주문 상품이 상품목록에 없는 상품으로 들어갔을때
    if product is None:
        return "상품을 찾을 수 없어요", 404

    # 주문 상품이 상품목록에 있으면 장바구니 추가
    item_name = product["name"]

    # 카트안에 같은이름이 있는 경우 수량만 증가시킴
    if item_name in py_cart:
        py_cart[item_name]["qty"] += 1
    
    # 카트에 최초  상품일때
    else:
        py_cart[item_name] = {"price":product["price"], "qty":1}

    # 변수 cart를 세션 "cart" key에 값으로 저장, 세션의 key는 여러개 만들 수 있다.
    session["se_cart"] = py_cart
    session.permanent = True #5분 만료 적용이 다시 시작

    # cart에 저장후 show_cart로 이동
    return redirect(url_for("show_cart"))

"""장바구니 부분 삭제"""
@app.route("/remove/<item_name>")
def remove_to_cart(item_name):
    py_cart = session.get("se_cart")

    if item_name in py_cart:
        del py_cart[item_name]

    session["se_cart"] = py_cart
    return redirect(url_for("show_cart"))
    # 라우팅에 수행되는 라우트핸들러함수이기 때문에  redirect가 있는거야
    # 서버에 있는데 크라우드를 통해서 부를거야
    # url_for에 요청에 대한 함수명을 써주면 app.route()를 
    # redirect를 사용해 url로 부르는 효과가 이루어짐

""" 장바구니 비우기 """
@app.get("/clear")
def clear_cart():
    # 세션의 여러개의 key중 "cart"키에 대한 내용을 몽땅 지움
    session.pop("se_cart", None)

    return redirect(url_for("show_cart"))


if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0")