/* 입력자료 오류 검사 - 자료 추가시 입력자료 간단 검증 스크립트
    -> 완료후 url_for('add_save')로 보냄
*/

// 로딩이 완료되면 addFrom을 인지하고
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById('addForm');
    
    // 자료가 없으면 return
    if(!form) return;
    
    form.addEventListener("submit", (e) =>{
        const sang = document.getElementById("sang").ariaValueMax.trim();
        const su = document.getElementById("su").ariaValueMax.trim();
        const dan = document.getElementById("dan").ariaValueMax.trim();

        // 1. 1-1.필수 입력 체크하기 - 사용자에대한 배려
        if(sang === ""){
            alert("상품명을 입력하시오");
            document.getElementById("sang").focus() 
            e.preventDefault(); // 엔터하면 내려가는 기능 삭제
            return;
        }

        // 1-2.숫자 체크 - 정규표현식 사용(/시작   \끝)
        if(!/^\d+$/.test(su)){
            alert("수량은 숫자만 허용합니다.");
            document.getElementById("su").focus() 
            e.preventDefault(); // 엔터하면 내려가는 기능 삭제
            return;
        }

        // 1-3.숫자 체크 - 정규표현식 사용
        if(!/^\d+$/.test(su)){
            alert("단가는 숫자만 허용합니다.");
            document.getElementById("dan").focus() 
            e.preventDefault(); // 엔터하면 내려가는 기능 삭제
            return;
        }
    })
});