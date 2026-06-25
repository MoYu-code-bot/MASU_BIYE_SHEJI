package com.hotpot.controller.admin;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.hotpot.common.PageResult;
import com.hotpot.common.Result;
import com.hotpot.dto.PageQuery;
import com.hotpot.entity.Reservation;
import com.hotpot.service.ReservationService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiImplicitParam;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;

@Api(tags = "B端-预订管理")
@RestController
@RequestMapping("/admin/reservations")
@RequiredArgsConstructor
public class AdminReservationController {

    private final ReservationService reservationService;

    @ApiOperation("分页查询预订")
    @GetMapping
    public Result<PageResult<Reservation>> page(PageQuery query,
                                                @RequestParam(required = false) Long storeId,
                                                @RequestParam(required = false) Integer status,
                                                @RequestParam(required = false) @DateTimeFormat(pattern = "yyyy-MM-dd") LocalDate date) {
        Page<Reservation> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<Reservation> wrapper = new LambdaQueryWrapper<>();
        if (storeId != null) {
            wrapper.eq(Reservation::getStoreId, storeId);
        }
        if (status != null) {
            wrapper.eq(Reservation::getStatus, status);
        }
        if (date != null) {
            wrapper.eq(Reservation::getReserveDate, date);
        }
        wrapper.orderByDesc(Reservation::getCreateTime);
        return Result.success(PageResult.of(reservationService.page(page, wrapper)));
    }

    @ApiOperation("确认预订")
    @ApiImplicitParam(name = "reservationId", value = "预订ID", required = true, dataType = "long", paramType = "query")
    @PutMapping("/confirm")
    public Result<?> confirm(@RequestParam Long reservationId) {
        reservationService.confirm(reservationId);
        return Result.success();
    }

    @ApiOperation("拒绝预订")
    @ApiImplicitParam(name = "reservationId", value = "预订ID", required = true, dataType = "long", paramType = "query")
    @PutMapping("/reject")
    public Result<?> reject(@RequestParam Long reservationId) {
        reservationService.reject(reservationId);
        return Result.success();
    }

    @ApiOperation("到店")
    @ApiImplicitParam(name = "reservationId", value = "预订ID", required = true, dataType = "long", paramType = "query")
    @PutMapping("/arrive")
    public Result<?> arrive(@RequestParam Long reservationId) {
        reservationService.arrive(reservationId);
        return Result.success();
    }

    @ApiOperation("完成预订")
    @ApiImplicitParam(name = "reservationId", value = "预订ID", required = true, dataType = "long", paramType = "query")
    @PutMapping("/complete")
    public Result<?> complete(@RequestParam Long reservationId) {
        reservationService.complete(reservationId);
        return Result.success();
    }
}
