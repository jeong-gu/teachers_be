let currentAudio = null;
let currentIcon = null;

function selectTeacher(name, photo) {
    localStorage.setItem("selectedTeacher", JSON.stringify({ name, photo }));
    window.location.href = "chat.html"; // chat.html로 이동
}

// 음성 아이콘 클릭 이벤트 리스너
document.querySelectorAll('.voice-icon').forEach(icon => {
    icon.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        
        const audioPath = this.getAttribute('data-audio');
        const teacherName = this.getAttribute('data-teacher');
        
        // 현재 재생 중인 오디오가 있으면 정지
        if (currentAudio && !currentAudio.paused) {
            currentAudio.pause();
            currentAudio.currentTime = 0;
            if (currentIcon) {
                currentIcon.classList.remove('playing');
            }
            
            // 같은 아이콘을 클릭했으면 정지만 하고 리턴
            if (currentIcon === this) {
                currentAudio = null;
                currentIcon = null;
                return;
            }
        }
        
        // 새 오디오 재생
        const audio = new Audio(audioPath);
        
        // 오디오 로드 에러 처리
        audio.onerror = function() {
            console.error(`음성 파일을 찾을 수 없습니다: ${audioPath}`);
            alert(`${teacherName} 선생님의 음성 파일을 찾을 수 없습니다.`);
        };
        
        // 재생 시작
        audio.play().then(() => {
            currentAudio = audio;
            currentIcon = this;
            this.classList.add('playing');
            
            // 재생 완료 시 애니메이션 제거
            audio.addEventListener('ended', () => {
                this.classList.remove('playing');
                currentAudio = null;
                currentIcon = null;
            });
            
        }).catch(error => {
            console.error('음성 재생 실패:', error);
            alert(`${teacherName} 선생님의 음성을 재생할 수 없습니다.`);
        });
    });
});

// 페이지를 떠날 때 오디오 정리
window.addEventListener('beforeunload', function() {
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }
});