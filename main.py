from fastapi import FastAPI, HTTPException
import numpy as np
import cv2
from typing import List, Dict, Tuple
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from itertools import permutations

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (필요에 따라 수정)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 노드 이름을 인덱스로 매핑
node_to_index = {}
MAX_NODES = 10
dist = np.full((MAX_NODES, MAX_NODES), np.inf)  # 거리 배열
next_node = np.full((MAX_NODES, MAX_NODES), -1)  # 경로 추적용 배열
node_count = 0  # 노드 수
INF = np.inf  # 오버플로 방지

# 노드 목록
nodes = ["o", "t1", "t2", "t3", "m", "m2", "b1", "b2", "b3"]

class Head:
    def __init__(self, id_: str, x1: int, y1: int, x2: int, y2: int):
        self.id = id_
        self.x1 = x1 * 10
        self.y1 = y1 * 10
        self.x2 = x2 * 10
        self.y2 = y2 * 10

    def calcul(self, image: np.ndarray) -> float:
        if image is None or image.size == 0:
            raise ValueError("Image is empty!")

        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        meanV = np.mean(img_hsv[:, :, 2])

        # Validate ROI
        if self.x1 < 0 or self.y1 < 0 or self.x2 > image.shape[1] or self.y2 > image.shape[0]:
            raise ValueError("Invalid ROI coordinates!")

        # Define ROI
        roi = img_hsv[self.y1:self.y2, self.x1:self.x2]
        mask1 = cv2.inRange(roi, (0, 0, 0), (255, 40, meanV * 0.7))
        mask2 = cv2.inRange(roi, (160, 0, 0), (200, 40, meanV * 0.7))
        blackPixels = cv2.countNonZero(mask1)

        # Compute ratio
        totalPixels = roi.size // 3
        blackRatio = blackPixels / totalPixels

        return blackRatio

    def check(self, image: np.ndarray, threshold: float) -> bool:
        if image is None or image.size == 0:
            raise ValueError("Cannot load image!")
        return self.calcul(image) > threshold

lis = {}

# 플로이드 와샬 알고리즘 함수
def floyd_warshall(edges: List[Tuple[str, Tuple[str, int]]]):
    global node_count
    for i in range(node_count):
        dist[i][i] = 0
        next_node[i][i] = i

    for src, (dst, weight) in edges:
        u = node_to_index[src]
        v = node_to_index[dst]
        dist[u][v] = weight
        dist[v][u] = weight
        next_node[u][v] = v
        next_node[v][u] = u

    for k in range(node_count):
        for i in range(node_count):
            for j in range(node_count):
                if dist[i][k] != INF and dist[k][j] != INF:
                    new_dist = dist[i][k] + dist[k][j]
                    if new_dist < dist[i][j]:
                        dist[i][j] = new_dist
                        next_node[i][j] = next_node[i][k]

# 두 노드 사이의 경로를 추적하는 함수
def reconstruct_path(start: int, end: int) -> List[str]:
    path = []
    while start != end:
        path.append(nodes[start])
        start = next_node[start][end]
    path.append(nodes[end])
    return path

# 지나쳐야 할 노드를 모두 지나 최단 거리를 계산하는 함수
def find_shortest_path_through_nodes(must_visit: List[str]) -> Tuple[int, str]:
    indices = [node_to_index[node] for node in must_visit]
    best_path = []
    min_distance = INF
    full_path_segments = []  # 전체 경로 저장

    # 순열 생성
    for perm in permutations(indices):
        current_distance = 0
        valid_path = True
        current_node = node_to_index["o"]  # 시작 노드

        # o -> 첫 노드
        next_node_index = perm[0]
        if dist[current_node][next_node_index] == INF:
            continue
        current_distance += dist[current_node][next_node_index]
        full_path_segments.append(reconstruct_path(current_node, next_node_index))

        # 각 노드 간 거리 합산
        for i in range(len(perm) - 1):
            u, v = perm[i], perm[i + 1]
            if dist[u][v] == INF:
                valid_path = False
                break
            current_distance += dist[u][v]
            full_path_segments.append(reconstruct_path(u, v))

        if not valid_path:
            continue

        # 마지막 노드 -> o
        last_node = perm[-1]
        if dist[last_node][node_to_index["o"]] == INF:
            continue
        current_distance += dist[last_node][node_to_index["o"]]
        full_path_segments.append(reconstruct_path(last_node, node_to_index["o"]))

        # 최단 거리 갱신
        if current_distance < min_distance:
            min_distance = current_distance
            best_path = perm  # 최단 경로 인덱스 저장

    # 결과 출력
    if min_distance == INF:
        return min_distance, "모든 노드를 지나 최단 거리를 찾을 수 없습니다."
    else:
        # 순찰 경로 생성
        howtogo = "o "  # 시작 노드
        current_node = node_to_index["o"]
        for idx in best_path:
            segment = reconstruct_path(current_node, idx)
            howtogo += " ".join(segment[1:]) + " "  # 첫 노드 중복 방지
            current_node = idx
        segment = reconstruct_path(current_node, node_to_index["o"])
        howtogo += " ".join(segment[1:])  # 마지막 노드에서 o로 돌아가는 경로 추가

        return min_distance, howtogo.strip()  # 최단 거리와 경로 반환

# 데이터 스키마 정의
class CheckData(BaseModel):
    id_: str
    image_path: str

class ImageData(BaseModel):
    image_path: str

@app.post("/check/")
async def allcheck(data: ImageData):
    global lis
    image_path = data.image_path

    print(f"Received Image Path: {image_path}")  # 디버깅용 출력
    img = cv2.imread(image_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Image not found or cannot be loaded")

    must_visit = []
    ateendence = {}
    people = 0
    print(lis)

    for id_, h in lis.items():
        if not h.check(img, 0.4):
            print(f"ID {h.id} is not present")  # 디버깅용 출력
            must_visit.append(h.id)
            ateendence[id_] = False
        else:
            print(f"ID {h.id} is present")  # 디버깅용 출력
            ateendence[id_] = True
            people += 1

    # 최단 거리 및 경로 계산
    if not must_visit:
        return {
            "min_distance": "0",
            "howtogo": "전원 출석",
            "ateendence": ateendence,
            "people": people
        }
    else:
        # 최단 경로 계산 함수 호출
        min_distance, howtogo = find_shortest_path_through_nodes(must_visit)
        return {
            "min_distance": min_distance,
            "howtogo": howtogo,
            "ateendence": ateendence,
            "people": people
        }

# POST 요청 처리
@app.post("/check_id/")
async def check_id(data: CheckData):  # pydantic 모델로 데이터 받기
    global lis
    id_ = data.id_
    image_path = data.image_path

    print(f"Received ID: {id_}, Image Path: {image_path}")  # 디버깅용 출력

    if id_ not in lis:
        raise HTTPException(status_code=404, detail="ID not found")

    img = cv2.imread(image_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Image not found or cannot be loaded")

    result = lis[id_].check(img, 0.4)
    return {"id": id_, "check_result": result}

# 서버 시작 시 실행되는 이벤트 핸들러
@app.on_event("startup")
async def startup_event():
    global node_count  # 전역 변수로 선언
    global lis
    # 노드 이름을 인덱스로 매핑
    for node in nodes:
        node_to_index[node] = node_count
        node_count += 1

    # 그래프 간선 정의
    edges = [
        ("o", ("t1", 13)), ("o", ("b1", 13)), ("t1", ("t2", 7)), ("t1", ("m", 5)),
        ("t1", ("b1", 7)), ("t2", ("m", 3)), ("t2", ("t3", 12)), ("t3", ("m2", 7)),
        ("m", ("m2", 7)), ("m2", ("b3", 7)), ("m2", ("b2", 7)), ("b1", ("b2", 7)),
        ("b2", ("b3", 4))
    ]

    # 플로이드 와샬 알고리즘 실행
    floyd_warshall(edges)

    # 예시 데이터 추가
    lis["a605"] = Head("b2", 25, 40, 53, 63)
    lis["a606"] = Head("b2", 92, 37, 104, 61)
    lis["a608"] = Head("b1", 77, 27, 86, 36)