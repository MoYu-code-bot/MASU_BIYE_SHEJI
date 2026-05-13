package com.captain.controller;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.captain.common.BaseRestController;
import com.captain.entity.po.Medical;
import com.captain.service.MedicalService;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 体检结果controller
 */
@RestController
@RequestMapping("/medical")
public class MedicalController extends BaseRestController<MedicalService, Medical> {

    @Override
    protected QueryWrapper<Medical> getSearchQueryWrapper(String searchValue, Medical medical) {
        QueryWrapper<Medical> wrapper = new QueryWrapper<>();
        if(StringUtils.isEmpty(wrapper)==false){
            wrapper.like("unit",searchValue)
                    .or().
                    like("stu_id",searchValue);
        }
        return wrapper;
    }
}
