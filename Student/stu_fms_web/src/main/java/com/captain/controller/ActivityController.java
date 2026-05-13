package com.captain.controller;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.captain.common.BaseRestController;
import com.captain.entity.po.Activity;
import com.captain.service.ActivityService;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 活动/竞赛的controller
 */
@RestController
@RequestMapping("/activity")
public class ActivityController extends BaseRestController<ActivityService, Activity> {

    @Override
    protected QueryWrapper<Activity> getSearchQueryWrapper(String searchValue, Activity activity) {
        QueryWrapper<Activity> wrapper = new QueryWrapper<>();
        if(StringUtils.isEmpty(searchValue)==false){
            wrapper.like("stu_id",searchValue)
                    .or()
                    .like("activity_name",searchValue);
        }
        return wrapper;
    }
}
