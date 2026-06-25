package com.hotpot.controller.api;

import com.hotpot.common.Result;
import com.hotpot.entity.Dish;
import com.hotpot.entity.Review;
import com.hotpot.entity.Store;
import com.hotpot.entity.TimeSlot;
import com.hotpot.service.DishService;
import com.hotpot.service.ReviewService;
import com.hotpot.service.StoreService;
import com.hotpot.service.TimeSlotService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiImplicitParam;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@Api(tags = "C端-门店接口")
@RestController
@RequestMapping("/api/stores")
@RequiredArgsConstructor
public class StoreApiController {

    private final StoreService storeService;
    private final DishService dishService;
    private final TimeSlotService timeSlotService;
    private final ReviewService reviewService;

    @ApiOperation("获取全部门店")
    @GetMapping("/list")
    public Result<List<Store>> list() {
        return Result.success(storeService.listAll());
    }

    @ApiOperation("门店详情")
    @ApiImplicitParam(name = "storeId", value = "门店ID", required = true, dataType = "long", paramType = "query")
    @GetMapping("/detail")
    public Result<Store> detail(@RequestParam Long storeId) {
        return Result.success(storeService.getById(storeId));
    }

    @ApiOperation("门店菜品")
    @ApiImplicitParam(name = "storeId", value = "门店ID", required = true, dataType = "long", paramType = "query")
    @GetMapping("/dishes")
    public Result<List<Dish>> dishes(@RequestParam Long storeId) {
        return Result.success(dishService.listByStoreId(storeId));
    }

    @ApiOperation("门店时段")
    @ApiImplicitParam(name = "storeId", value = "门店ID", required = true, dataType = "long", paramType = "query")
    @GetMapping("/slots")
    public Result<List<TimeSlot>> slots(@RequestParam Long storeId) {
        return Result.success(timeSlotService.listByStoreId(storeId));
    }

    @ApiOperation("门店评价")
    @GetMapping("/reviews")
    public Result<List<Review>> reviews(@RequestParam Long storeId,
                                        @RequestParam(defaultValue = "1") Integer pageNum,
                                        @RequestParam(defaultValue = "10") Integer pageSize) {
        return Result.success(reviewService.listByStoreId(storeId, pageNum, pageSize));
    }
}
