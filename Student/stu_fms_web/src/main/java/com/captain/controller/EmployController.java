package com.captain.controller;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.captain.common.BaseRestController;
import com.captain.entity.po.Employ;
import com.captain.service.EmployService;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 就业信息controller
 */
@RestController
@RequestMapping("/employ")
public class EmployController extends BaseRestController<EmployService, Employ> {

    @Override
    protected QueryWrapper<Employ> getSearchQueryWrapper(String searchValue, Employ employ) {
        QueryWrapper<Employ> wrapper = new QueryWrapper<>();
        if(StringUtils.isEmpty(wrapper)==false){
            wrapper.like("stu_id",searchValue)
                    .or()
                    .like("company",searchValue);
        }
        return wrapper;
    }
}
