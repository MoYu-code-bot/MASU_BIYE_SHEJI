package com.captain.controller;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.captain.common.BaseRestController;
import com.captain.entity.po.Course;
import com.captain.entity.po.StuInfo;
import com.captain.service.CourseService;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 课程的controller
 */
@RestController
@RequestMapping("/course")
public class CourseController extends BaseRestController<CourseService, Course> {

    @Override
    protected QueryWrapper<Course> getSearchQueryWrapper(String searchValue, Course course) {
        if(!StringUtils.isEmpty(searchValue)){
            QueryWrapper<Course> wrapper=new QueryWrapper();
            wrapper.like("course_name",searchValue);
            return wrapper;
        }else{
            return super.getSearchQueryWrapper(searchValue, course);
        }
    }
}
