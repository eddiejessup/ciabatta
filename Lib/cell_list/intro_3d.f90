module cell_list_3d

    use utils
    implicit none

! dim
    integer, allocatable, private :: cl(:, :, :, :), cli(:, :, :)
    integer, parameter, private :: m_max = 500
! /dim
    integer, allocatable :: inters(:, :), intersi(:)

contains

subroutine initialise_cl(n, m)
    integer, intent(in) :: n, m

    if (allocated(cl) .and. (size(cl, 1) /= n .or. size(cl, 2) /= m)) then
        deallocate(cl)
        deallocate(cli)
    end if
    if (.not. allocated(cl)) then
! dim
        allocate(cl(n, m, m, m))
        allocate(cli(size(cl, 2), size(cl, 3), size(cl, 4)))
! /dim
    end if
end subroutine

subroutine initialise_inters(n)
    integer, intent(in) :: n

    if (allocated(inters) .and. size(inters, 1) /= n) then
        deallocate(inters)
        deallocate(intersi)
    end if
    if (.not. allocated(inters)) then
        allocate(inters(n, n))
        allocate(intersi(size(inters, 2)))
    end if
end subroutine

subroutine make_inters(r, l, r_cut)
    real(dp), intent(in) :: r(:, :), l, r_cut
! dim
    integer :: inds(size(r, 1), size(r, 2)), m, x, y, z, x_inc, x_dec, y_inc, y_dec, z_inc, z_dec, i, i_cl
! /dim
    real(dp) :: l_half, r_cut_sq

    m = min(int(floor(l / r_cut)), m_max)
    l_half = l / 2.0_dp
    r_cut_sq = r_cut ** 2
    call initialise_cl(size(r, 2), m)
    call initialise_inters(size(r, 2))

    inds = r_to_inds(r, l, m)

    cli = 0
    do i = 1, size(r, 2)
! dim
        cli(inds(1, i), inds(2, i), inds(3, i)) = cli(inds(1, i), inds(2, i), inds(3, i)) + 1
        cl(cli(inds(1, i), inds(2, i), inds(3, i)), inds(1, i), inds(2, i), inds(3, i)) = i
    end do

    intersi = 0
    do x = 1, m
        x_inc = i_wrap(x + 1, m)
        x_dec = i_wrap(x - 1, m)
        do y = 1, m
            y_inc = i_wrap(y + 1, m)
            y_dec = i_wrap(y - 1, m)
            do z = 1, m
                z_inc = i_wrap(z + 1, m)
                z_dec = i_wrap(z - 1, m)
                do i_cl = 1, cli(x, y, z)
                    i = cl(i_cl, x, y, z)

                    call core(i, x, y, z)
                    call core(i, x_inc, y, z)
                    call core(i, x_dec, y, z)

                    call core(i, x, y_inc, z)
                    call core(i, x_inc, y_inc, z)
                    call core(i, x_dec, y_inc, z)

                    call core(i, x, y_dec, z)
                    call core(i, x_inc, y_dec, z)
                    call core(i, x_dec, y_dec, z)


                    call core(i, x, y, z_inc)
                    call core(i, x_inc, y, z_inc)
                    call core(i, x_dec, y, z_inc)

                    call core(i, x, y_inc, z_inc)
                    call core(i, x_inc, y_inc, z_inc)
                    call core(i, x_dec, y_inc, z_inc)

                    call core(i, x, y_dec, z_inc)
                    call core(i, x_inc, y_dec, z_inc)
                    call core(i, x_dec, y_dec, z_inc)


                    call core(i, x, y, z_dec)
                    call core(i, x_inc, y, z_dec)
                    call core(i, x_dec, y, z_dec)

                    call core(i, x, y_inc, z_dec)
                    call core(i, x_inc, y_inc, z_dec)
                    call core(i, x_dec, y_inc, z_dec)

                    call core(i, x, y_dec, z_dec)
                    call core(i, x_inc, y_dec, z_dec)
                    call core(i, x_dec, y_dec, z_dec)
                end do

            end do
        end do
    end do

contains

    subroutine core(i, x, y, z)
        integer, intent(in) :: i, x, y, z

        integer :: i_target_cl, i_target
        do i_target_cl = 1, cli(x, y, z)
            i_target = cl(i_target_cl, x, y, z)
! /dim
            if (i_target /= i .and. r_sep_sq(r(:, i), r(:, i_target), l, l_half) < r_cut_sq) then
                intersi(i) = intersi(i) + 1
                inters(intersi(i), i) = i_target
            end if
        end do
        return
    end subroutine

end subroutine

end module