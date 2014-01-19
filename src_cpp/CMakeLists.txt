cmake_minimum_required(VERSION 2.8)
project(anpr)
set(detect_plate_region_src detect_plates.cc main.cc utility.cc)
find_package(OpenCV REQUIRED)
add_executable(detect_plate ${detect_plate_region_src})
target_link_libraries(detect_plate ${OpenCV_LIBS})

add_executable(parse_chars parse_chars.cc parse_chars_test.cc utility.cc)
target_link_libraries(parse_chars ${OpenCV_LIBS})

add_executable(judge_chars judge_chars.cc judge_chars_test.cc parse_chars.cc utility.cc)
target_link_libraries(judge_chars ${OpenCV_LIBS})