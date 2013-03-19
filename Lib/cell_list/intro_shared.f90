module cell_list_shared

    use utils
    implicit none

    integer, allocatable :: inters(:, :), intersi(:)

contains

subroutine initialise_inters(n)
    integer, intent(in) :: n

    if (allocated(inters)) deallocate(inters)
    if (allocated(intersi)) deallocate(intersi)
    if (.not. allocated(inters)) allocate(inters(n, n))
    if (.not. allocated(intersi)) allocate(intersi(size(inters, 2)))
end subroutine

end module
