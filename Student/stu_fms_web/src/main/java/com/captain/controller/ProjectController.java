package com.captain.controller;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.captain.common.BaseRestController;
import com.captain.entity.po.Project;
import com.captain.service.ProjectService;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 科研项目controller
 */
@RestController
@RequestMapping("/project")
public class ProjectController extends BaseRestController<ProjectService, Project> {

    @Override
    protected QueryWrapper<Project> getSearchQueryWrapper(String searchValue, Project project) {
        QueryWrapper<Project> wrapper=new QueryWrapper<>();
        if(StringUtils.isEmpty(searchValue)==false){
            wrapper.like("pro_name",searchValue);
        }
        return wrapper;
    }
}
