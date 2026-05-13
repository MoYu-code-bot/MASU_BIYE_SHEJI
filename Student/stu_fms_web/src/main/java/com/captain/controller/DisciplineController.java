package com.captain.controller;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.captain.common.BaseRestController;
import com.captain.entity.po.Discipline;
import com.captain.service.DisciplineService;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 违纪controller
 */
@RestController
@RequestMapping("/discipline")
public class DisciplineController extends BaseRestController<DisciplineService, Discipline> {

    @Override
    protected QueryWrapper<Discipline> getSearchQueryWrapper(String searchValue, Discipline discipline) {
        QueryWrapper<Discipline> wrapper = new QueryWrapper<>();
        if(StringUtils.isEmpty(searchValue)==false){
            wrapper.like("stu_id",searchValue)
                    .or()
                    .like("dis_name",searchValue);
        }
        return wrapper;
    }
}
