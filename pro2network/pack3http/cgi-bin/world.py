s1 = "자료1"
s2 = "두번째 자료"
"""
MIME type : 내가 넘겨주는 문서의 성격은 이런거야~
MIME타입 종료 : https://developer.mozilla.org/ko/docs/Web/HTTP/Guides/MIME_types/Common_types
브라우저는 등록되어있는 마인타입이 있음
등록되어있지 않은 마인타입을 받으면 다운로드 시킴
get 방식은 길이의 제한(256자), 보안의 문제가 있다.
"""
print("Content-Type:text/html;charset:utf-8\n")
print("""
    <html lang="kr">
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>world</title>
    </head>
    <body>
        <h1>world 페이지</h1>
        자료출력 : {0}, {1}
        <br/>
        <img src="../images/dog.jpeg">
        <br/>
        <a href="../index.html">메인으로</a>
    </body>
    </html>
        """.format(s1, s2)) # 템플레이트.

# 경로 변경: ../images/dog.jpeg => 상위폴더로 올라가서/images/dog.jpeg