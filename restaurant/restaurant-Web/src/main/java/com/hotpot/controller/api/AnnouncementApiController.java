package com.hotpot.controller.api;

import com.hotpot.common.Result;
import com.hotpot.entity.Announcement;
import com.hotpot.service.AnnouncementService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiImplicitParam;
import io.swagger.annotations.ApiOperation;
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

    @ApiOperation("获取公告列表")
    @ApiImplicitParam(name = "storeId", value = "门店ID", required = true, dataType = "long", paramType = "query")
    @GetMapping("/list")
    public Result<List<Announcement>> list(@RequestParam Long storeId) {
        return Result.success(announcementService.listByStoreId(storeId));
    }
}
