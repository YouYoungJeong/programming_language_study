/* 클라이언트 사이드 */
// 함수(화살표 함수) 객체 생성 후  $에 할당
// $ : 함수를 담고있는 상수
// alert("a");

// 현재 작업중인 객체가 들어오면 그 객체가 선택됨 그 객체는 $가 가지고 있어.- 동적
const $ = (sel) => document.querySelector(sel);
// ===
// function $(sel){
//     return document.querySelector(sel);
// }
// ex) $("#sendBtn") 하면($호출) document.querySelector(sel)가 실행이 됨.

$("#sendBtn").addEventListener("click", async () => { // 비동기 처리
    const name = $('#name').value.trim();
    const age = $('#age').value.trim();
    //  ===
    // const age = document.querySelector('#age').value.trim();
    
    // new URLSearchParams() : 길동 -> %ED%78...공백, 한글 등이 포함된 경우 자동 인코딩
    const params = new URLSearchParams({name, age}); 
    //=== key와 value가 같은면 하나씩 사용 가능해
    // const params = {name=name, age=age};

    //최종 URL생성(중급) -> 넘어가는 모양: /api/friend?name=%ED%78(길동)&age=23
    const url = `/api/friend?${params.toString()}`;
    
    /*서버에 자료 요청시간이 길어지면 보이는 메세지
    현재 내컴퓨터로만 진행해서 볼일은 없지만 네트워크로 넘어가면 필요*/
    $("#result").textContent = "요청 중.." 
    // ===
    // document.querySelector("#result")

    // 네트워크가 나오면 무조건 try가 나와!
    try{
        const res = await fetch(url,{ // url을 파라미터2개를 들고 요청하고있어
            method:"GET",
            headers:{"Accept":"application/json"} //서버 에게 요청, MIME type :데이터의 형식을 명시하는 표준 규격
        });

        /* 모양은 JSON형태로 {key:value} incoding해서 문자열로 넘어가지만, 
        우리는 JSON의 형태 객체 받고싶어서 요청하는거야
        ajax로 부분데이터 받아오기*/
        const data = await res.json(); // 응답 본문을 JSON으로 파싱해서 JS객체화
        if(!res.ok || data.ok === false){
            $("#result").innerHTML = `<span class='error'> err : ${data.error} </span>`;
            return; // err없으면 함수 빠져나가
        }
        //요청 성공
        $("#result").innerHTML = `
            <div>이름 : ${data.name}</div>
            <div>나이 : ${data.age}</div>
            <div>연령대 : ${data.age_group}</div>
            <div>메세지 : ${data.message}</div>
        `;

        /*  $("$result") : 우리가 만든 $ / 
            ${err} ``에서 매핑을 위한 $ - 문법 : 서로 다름!*/
    }catch(err){
        $("#result").innerHTML = `<span class='error'>네트워크, 파싱 오류 : ${err} </span>`;
    }


});
