package com.hotpot.controller.api;

import com.hotpot.common.Result;
import com.hotpot.entity.Announcement;
import com.hotpot.service.AnnouncementService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiParam;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@Api(tags = "C端-公告接口")
@RestController
@RequestMapping("/api/announcements")
@RequiredArgsConstructor
public class AnnouncementApiController {

    private final AnnouncementService announcementService;

    @GetMapping("list")
    @ApiOperation("获取公告列表")
    public Result<List<Announcement>> list(@ApiParam("门店ID") @RequestParam(required = false) Long storeId) {
        if (storeId != null) {
            return Result.success(announcementService.listByStoreId(storeId));
        }
        return Result.success(announcementService.listActive());
    }
}
